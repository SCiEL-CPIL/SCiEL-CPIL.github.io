"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_bound(f):
    """Compute the autoconvolution upper bound (max autocorrelation / N)."""
    # Compute autocorrelation via FFT for efficiency
    n = len(f)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    acf = np.fft.irfft(F * np.conj(F), n=fft_size)[:n]
    # The bound is the maximum absolute autocorrelation divided by N
    return np.max(np.abs(acf)) / n

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    best_score = None
    best_f = None
    
    # Try different low-frequency sine wave patterns
    # sign(sin(2*pi*k*x/N)) for various k
    x = np.arange(N)
    
    for k in range(1, 40):
        if time.time() > deadline:
            break
        
        # Create sign pattern from sine wave
        phase = 2 * np.pi * k * x / N
        f = np.sign(np.sin(phase))
        # Replace zeros with 1 to ensure no zero values
        f[f == 0] = 1
        
        score = _autocorr_bound(f)
        
        if best_score is None or score < best_score:
            best_score = score
            best_f = f.copy()
    
    # If no valid result found, return ones as fallback
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
