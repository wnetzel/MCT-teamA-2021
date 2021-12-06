"""
This package extracts power and frequency characteristics of a signal.
"""
import numpy as np

def root_mean_square(audio, frame_length, hop_length):

    """
    Compute root-mean-square (RMS) value from scratch with the following formula, 
    for each frame of the audio samples.
    """
    
    # Pad with the reflection of the signal so that the frames are centered 
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples     
    signal = np.pad(audio, int(frame_length // 2), mode='reflect')
    
    rms = []
    
    for i in range(0, audio.shape[0], hop_length):
        
        rms_formula = np.sqrt(1 / frame_length * np.sum(signal[i:i+frame_length]**2))        
        rms.append(rms_formula)
        
    return np.array(rms)  

def spectral_centroid(audio, frame_length, hop_length, sr=44100):

    """
    As the name suggests, a spectral centroid is the location of the centre of mass of the spectrum. 
    Since the audio files are digital signals and the spectral centroid is a measure that can be useful 
    in the characterization of the spectrum of the audio file signal.
    """
    
    # Pad with the reflection of the signal so that the frames are centered 
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(audio, int(frame_length // 2), mode='reflect')
    
    centroid = []
    
    for i in range(0, audio.shape[0], hop_length):
        
        cent = signal[i:i+frame_length]
            
        # Compute the discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT)
        magnitudes = np.abs(np.fft.fft(cent)) # magnitude of absolute (real) frequency values
        
        # Compute only the positive half of the DFT (i.e 1 + first half)
        mag = magnitudes[:int(1 + len(magnitudes) // 2)]
        
        # Compute the center frequencies of each bin
        freq = np.linspace(0, sr/2, int(1 + len(cent) // 2)) 
        
        # Return weighted mean of the frequencies present in the signal
        normalize_mag = mag / np.sum(mag)
        centroid.append(np.sum(freq * normalize_mag)) 
        
    return np.array(centroid)

def spectral_bandwidth(audio, frame_length, hop_length, sr=44100, p=2):

    """
    Bandwidth is the difference between the upper and lower frequencies in a continuous band of frequencies. 
    As we know the signals oscillate about a point so if the point is the centroid of the signal then the sum of maximum 
    deviation of the signal on both sides of the point can be considered as the bandwidth of the signal at that time frame.
    """
    
    # Pad with the reflection of the signal so that the frames are centered 
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(audio, int(frame_length // 2), mode='reflect')
    
    bandwidth = []
    
    for i in range(0, audio.shape[0], hop_length):
        
        frame = signal[i:i+frame_length]
            
        # Compute the discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT)
        magnitudes = np.abs(np.fft.fft(frame)) # magnitude of absolute (real) frequency values
        
        # Compute only the positive half of the DFT (i.e 1 + first half)
        mag = magnitudes[:int(1 + len(magnitudes) // 2)]
        
        # Compute the center frequencies of each bin
        freq = np.linspace(0, sr/2, int(1 + len(frame) // 2))
        
        # Return weighted mean of the frequencies present in the signal
        normalize_mag = mag / np.sum(mag)
        centroid = np.sum(freq * normalize_mag)
        
        spectral_bandwidth = np.sum(normalize_mag * abs(freq - centroid) ** p) ** (1.0/p)
        bandwidth.append(spectral_bandwidth) 
        
    return np.array(bandwidth)