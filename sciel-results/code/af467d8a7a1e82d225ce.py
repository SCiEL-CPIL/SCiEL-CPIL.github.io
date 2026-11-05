import time
import math
import numpy as np

def _compute_lpf_sieve(upper):
    """Compute largest prime factor for all integers in [0, upper] using a sieve."""
    lpf = np.zeros(upper + 1, dtype=np.int32)
    for i in range(2, upper + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
    return lpf

def construct():
    deadline = time.time() + 80
    upper = 1000001
    lower = 10000

    # Precompute largest prime factor for all numbers up to upper
    lpf = _compute_lpf_sieve(upper)

    # Compute max(lpf[n], lpf[n+1]) for n in [lower, 1000000]
    lpf_n = lpf[lower:upper]
    lpf_np1 = lpf[lower+1:upper+1]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the candidate with minimum max LPF
    min_idx = int(np.argmin(max_lpf))
    best_n = lower + min_idx
    best_val = int(max_lpf[min_idx])
    
    # If we have time, do a local refinement around the best candidate
    # to verify and potentially find nearby better candidates
    if time.time() < deadline - 5:
        # Search a window around the best candidate for verification
        window = 100
        start = max(lower, best_n - window)
        end = min(1000000, best_n + window)
        
        local_best_n = best_n
        local_best_val = best_val
        
        for n in range(start, end + 1):
            v = max(int(lpf[n]), int(lpf[n+1]))
            if v < local_best_val:
                local_best_val = v
                local_best_n = n
        
        best_n = local_best_n
        best_val = local_best_val

    return int(best_n)
