"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_autocorr_max_abs(x):
    """Compute max absolute value of linear autocorrelation using FFT."""
    n = len(x)
    # Pad to next power of 2 for efficient FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conj(f), size)[:n]
    return np.max(np.abs(acf))

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin
    
    # Start with random initial configuration
    x = np.random.choice([-1, 1], N).astype(np.int8)
    best_val = _fast_autocorr_max_abs(x)
    best_x = x.copy()
    
    # Simulated Annealing
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-4
    
    while time.time() < deadline and T > min_T:
        # Pick a random position to flip
        i = random.randint(0, N - 1)
        x[i] = -x[i]
        new_val = _fast_autocorr_max_abs(x)
        
        # Accept if better, or with probability exp(-delta/T) if worse
        delta = new_val - best_val
        if delta < 0 or random.random() < np.exp(-delta / T):
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] = -x[i]  # Undo
        
        T *= cooling_rate
    
    return best_x
# EVOLVE-BLOCK-END
