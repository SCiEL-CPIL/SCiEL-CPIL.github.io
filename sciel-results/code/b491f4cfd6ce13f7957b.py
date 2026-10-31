"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Strategy: Chirp signal
    # x_k = cos(2*pi*alpha*k^2)
    # We want to minimize max|autocorr| / sum(x^2)
    
    # Try a few alpha values
    best_x = np.ones(N)
    best_score = np.inf
    
    # Autocorrelation via FFT
    def calc_score(x):
        # Linear autocorrelation
        # Pad to 2N to avoid circular convolution
        n_fft = 1 << (2*N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        AC = np.fft.irfft(X * np.conj(X), n=n_fft)
        # AC is real, length n_fft. 
        # Linear autocorrelation is in AC[0:N] and AC[N-1::-1]
        # Actually, np.fft.irfft of X*conj(X) gives circular autocorrelation of padded x.
        # Since x is zero-padded, it's linear.
        # The peak is at 0.
        max_ac = np.max(np.abs(AC[:N])) # Only first half needed for symmetry? 
        # AC is symmetric. AC[k] = AC[N-k] in circular? No.
        # Linear autocorr c_k = sum x_i x_{i+k}.
        # FFT method: IFFT(|FFT(x)|^2).
        # The result is circular autocorr of x padded to n_fft.
        # For k in 0..N-1, it is the linear autocorr.
        energy = np.sum(x**2)
        if energy == 0:
            return np.inf
        return max_ac / energy

    # Generate chirps
    k = np.arange(N)
    
    # Test different chirp rates
    # alpha = 1/(2N) is standard
    candidates = []
    for i in range(1, 10):
        alpha = i / (2.0 * N)
        x = np.cos(2.0 * np.pi * alpha * k**2)
        candidates.append(x)
        
    # Also try linear phase (less optimal but check)
    # x = np.cos(2.0 * np.pi * k / N)
    
    for x in candidates:
        s = calc_score(x)
        if s < best_score:
            best_score = s
            best_x = x
        if time.time() > deadline:
            break
            
    return best_x
# EVOLVE-BLOCK-END
