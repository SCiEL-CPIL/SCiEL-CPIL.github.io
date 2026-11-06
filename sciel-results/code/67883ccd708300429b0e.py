import numpy as np
import time

def _build_lpf_sieve(n):
    lpf = np.zeros(n + 1, dtype=np.int32)
    lpf[1] = 1
    for i in range(2, n + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
    return lpf

def construct():
    target = 123457
    lpf = _build_lpf_sieve(target)
    
    # a ranges from 1 to target // 2
    # b = target - a
    # We want min of max(lpf[a], lpf[b])
    
    # Slices
    # a indices: 1 to target//2
    # b indices: target-1 down to target - target//2
    
    a_vals = np.arange(1, target // 2 + 1)
    b_vals = target - a_vals
    
    pa = lpf[a_vals]
    pb = lpf[b_vals]
    
    max_pfs = np.maximum(pa, pb)
    
    min_idx = np.argmin(max_pfs)
    min_val = max_pfs[min_idx]
    
    best_a = a_vals[min_idx]
    best_b = b_vals[min_idx]
    
    return [int(best_a), int(best_b)]
