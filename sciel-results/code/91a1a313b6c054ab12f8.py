"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _value(h):
    S = np.sum(h)
    if S == 0:
        return float('inf')
    c = np.convolve(h, h)
    return abs(2 * N * np.max(c) / (S ** 2))

def construct():
    start = time.time()
    deadline = start + 110  # safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try square wave modes with different periods and amplitudes
    # h[i] = 1 + a * s[i], where s has zero mean
    for period in [2, 4, 6, 8, 10, 12, 16, 20]:
        if time.time() > deadline:
            break
        # Create square wave with given period, zero mean
        s = np.zeros(N)
        half = period // 2
        for i in range(N):
            s[i] = 1.0 if (i % period) < half else -1.0
        # Ensure zero mean (adjust if needed)
        s -= np.mean(s)
        
        # Optimize a in a range
        a_vals = np.linspace(-2, 2, 100)
        for a in a_vals:
            if time.time() > deadline:
                break
            h = 1.0 + a * s
            val = _value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
