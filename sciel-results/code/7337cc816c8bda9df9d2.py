import numpy as np
import time
from scipy.signal import fftconvolve

N = 400
DEADLINE = 145.0  # seconds, with safety margin

def _compute_c3prime(x):
    """Compute C3' = max(|x*x|) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    conv = fftconvolve(x, x, mode='full')
    return np.max(np.abs(conv)) / (s * s)

def construct():
    """
    Constructs a signed step function (±1 values) minimizing C3'.
    Uses local search to optimize the sign pattern.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Start with constant signal
    best_x = np.ones(N)
    best_score = _compute_c3prime(best_x)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_x
    
    # Local search on ±1 sequences
    for init_m in [1, 2, 5, 10, 20, 50]:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            x = np.ones(N)
            if init_m > 0:
                flip_idx = rng.choice(N, size=init_m, replace=False)
                x[flip_idx] = -1
            
            score = _compute_c3prime(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
            
            # Local search: flip individual bits
            for iteration in range(3000):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                idx = rng.integers(0, N)
                x_new = x.copy()
                x_new[idx] = -x_new[idx]
                
                s_new = np.sum(x_new)
                if abs(s_new) < 1e-15:
                    continue
                
                conv_new = fftconvolve(x_new, x_new, mode='full')
                score_new = np.max(np.abs(conv_new)) / (s_new * s_new)
                
                if score_new < score:
                    score = score_new
                    x = x_new
                    if score < best_score:
                        best_score = score
                        best_x = x_new.copy()
    
    return best_x
