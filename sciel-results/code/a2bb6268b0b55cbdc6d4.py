import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    n = 30
    half = 15
    
    # Calculate LCM(1..30)
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // (i + 1) for i in range(n)]
    
    w1 = weights[:half]
    w2 = weights[half:]
    
    size = 3 ** half
    
    def generate_sums(weights_half):
        idx = np.arange(size, dtype=np.int64)
        sums = np.zeros(size, dtype=np.int64)
        
        for j in range(half):
            power = 3 ** j
            digit = (idx // power) % 3
            coeff = digit - 1  # -1, 0, 1
            sums += coeff * weights_half[j]
            
        return sums

    if time.time() - start_time > timeout * 0.1:
        return [0]*28 + [1, -1]

    try:
        sums1 = generate_sums(w1)
        sums2 = generate_sums(w2)
        
        # Sort sums2
        sorted_indices2 = np.argsort(sums2)
        sorted_sums2 = sums2[sorted_indices2]
        
        targets = -sums1
        
        idxs = np.searchsorted(sorted_sums2, targets)
        
        idxs_c1 = idxs
        idxs_c2 = idxs - 1
        
        large_val = np.iinfo(np.int64).max
        
        s2_cand1 = np.full(len(sums1), large_val, dtype=np.int64)
        s2_cand2 = np.full(len(sums1), large_val, dtype=np.int64)
        
        valid_c1 = idxs_c1 < len(sorted_sums2)
        valid_c2 = idxs_c2 >= 0
        
        s2_cand1[valid_c1] = sorted_sums2[idxs_c1[valid_c1]]
        s2_cand2[valid_c2] = sorted_sums2[idxs_c2[valid_c2]]
        
        total1 = sums1 + s2_cand1
        total2 = sums1 + s2_cand2
        
        valid_total1 = valid_c1 & (total1 != 0)
        valid_total2 = valid_c2 & (total2 != 0)
        
        abs_total1 = np.abs(total1)
        abs_total2 = np.abs(total2)
        
        abs_total1[~valid_total1] = large_val
        abs_total2[~valid_total2] = large_val
        
        min1 = np.min(abs_total1)
        min2 = np.min(abs_total2)
        
        if min1 < min2:
            best_diff = min1
            best_idx1 = np.argmin(abs_total1)
            best_idx2 = sorted_indices2[idxs_c1[best_idx1]]
        else:
            best_diff = min2
            best_idx1 = np.argmin(abs_total2)
            best_idx2 = sorted_indices2[idxs_c2[best_idx1]]
            
        def idx_to_coeffs(idx, half_len):
            coeffs = [0] * half_len
            temp = idx
            for j in range(half_len):
                digit = temp % 3
                temp //= 3
                coeffs[j] = digit - 1
            return coeffs
        
        c1 = idx_to_coeffs(best_idx1, half)
        c2 = idx_to_coeffs(best_idx2, half)
        
        return c1 + c2

    except Exception:
        return [0]*28 + [1, -1]

# EVOLVE-BLOCK-END
