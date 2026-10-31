import numpy as np

N = 100

def construct():
    # Strategy: Formulate the problem in the Fourier domain.
    # The maximum overlap is minimized when the autocorrelation C_t is constant.
    # This occurs when the power spectrum |H_k|^2 is zero for all k != 0.
    # This implies h is constant.
    # Given the sum constraint sum(h) = N/2, the constant value is 0.5.
    
    # Construct the optimal Fourier coefficients
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0  # DC component
    
    # Compute inverse FFT to get h in time domain
    h = np.fft.ifft(H).real
    
    # Enforce bounds [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Ensure sum is exactly N/2
    # Due to floating point precision, adjust slightly if needed
    s = np.sum(h)
    if s != 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    
    # Final exact sum correction
    diff = N / 2.0 - np.sum(h)
    if diff != 0:
        # Distribute the difference to the first element
        # Since h is 0.5, there is plenty of room in [0, 1]
        h[0] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h
