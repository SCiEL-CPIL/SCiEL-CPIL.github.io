"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_bound(signs):
    """Compute the upper bound on C3 (max absolute autoconvolution value).
    
    For a step function with signs on N intervals over [-1/4, 1/4],
    the autoconvolution at lag k (in interval units) is the sum of 
    signs[i] * signs[i-k] for valid i, normalized by N.
    
    We want to minimize the max |autocorrelation| over non-zero lags.
    """
    signs = np.asarray(signs, dtype=np.float64)
    n = len(signs)
    
    # Compute autocorrelation for all lags
    # autocorr[k] = sum_{i} signs[i] * signs[i+k] / n for k >= 0
    # Use FFT for efficiency
    # Pad to next power of 2 for efficient FFT
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    f = np.zeros(nfft)
    f[:n] = signs
    
    F = np.fft.fft(f)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft(power).real
    autocorr /= n  # normalize
    
    # autocorr[k] for k=0..n-1, and autocorr[nfft-k] for negative lags
    # We want max |autocorr[k]| for k != 0
    # autocorr[0] is always 1 (since signs are ±1, sum of squares = n)
    
    # Get all non-zero lags
    max_val = 0.0
    for k in range(1, n):
        v = abs(autocorr[k])
        if v > max_val:
            max_val = v
        # Also check the negative lag equivalent
        v2 = abs(autocorr[nfft - k])
        if v2 > max_val:
            max_val = v2
    
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    # Start with random signs
    best_signs = np.random.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_bound(best_signs)
    
    # Simulated annealing / local search
    current_signs = best_signs.copy()
    current_score = best_score
    
    # Parameters
    T = 0.1  # Initial temperature
    cooling = 0.9995
    
    iteration = 0
    while time.time() < deadline:
        # Try flipping a random position
        idx = np.random.randint(0, N)
        new_signs = current_signs.copy()
        new_signs[idx] *= -1
        
        new_score = _autoconv_bound(new_signs)
        
        # Accept if better, or with probability exp(-(delta)/T)
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_signs = new_signs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
        
        T *= cooling
        iteration += 1
        
        # Check time periodically
        if iteration % 1000 == 0:
            if time.time() > deadline:
                break
    
    return best_signs
# EVOLVE-BLOCK-END
