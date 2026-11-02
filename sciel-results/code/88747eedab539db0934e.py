"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(f):
    """Compute max absolute value of linear autoconvolution."""
    # Use FFT for speed
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * np.conj(F), size)
    # The linear autoconv is ac[0:n] (lags 0 to n-1)
    return np.max(np.abs(ac[:n]))

def _integral_sq(f):
    """Squared integral = (sum f)^2"""
    return (np.sum(f)) ** 2

def _score(f):
    """C3' = max|autoconv| / (sum f)^2"""
    num = _autoconv_max_abs(f)
    den = _integral_sq(f)
    if den == 0:
        return float('inf')
    return num / den

def construct():
    deadline = time.time() + 140  # 140s safety margin for 150s kill
    best_f = np.ones(N, dtype=np.float64)
    best_score = _score(best_f)
    
    # Local search: flip random bits, keep if improves
    f = best_f.copy()
    current_score = best_score
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try flipping one random position
        idx = random.randint(0, N - 1)
        f[idx] *= -1  # flip sign
        new_score = _score(f)
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            f[idx] *= -1  # revert
        
        # Occasionally try flipping a small block
        if random.random() < 0.1:
            block_len = random.randint(2, 5)
            start = random.randint(0, N - block_len)
            f[start:start+block_len] *= -1
            new_score = _score(f)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                f[start:start+block_len] *= -1  # revert
    
    # Return as signed step heights
    return best_f
# EVOLVE-BLOCK-END
