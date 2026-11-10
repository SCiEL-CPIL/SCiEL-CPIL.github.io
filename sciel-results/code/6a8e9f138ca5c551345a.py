import numpy as np
from math import gcd
from functools import reduce
import time

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    # 1. Calculate LCM of 1..30
    L = _lcm_list(range(1, 31))
    
    # 2. Calculate weights W_i = L / i
    weights = np.array([L // (i + 1) for i in range(30)], dtype=np.int64)
    
    # 3. Split into two halves
    n = 30
    n1 = n // 2
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    def generate_sums_and_codes(weights_half):
        """
        Generates all possible sums and their base-3 encoded coefficient vectors.
        """
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int32)
        
        for j, w in enumerate(weights_half):
            power = 3 ** j
            
            # Branch 0: coeff = -1 (code digit 0)
            # Branch 1: coeff = 0  (code digit 1)
            # Branch 2: coeff = 1  (code digit 2)
            
            s_keep = sums           # coeff 0
            s_add  = sums + w       # coeff 1
            s_sub  = sums - w       # coeff -1
            
            c_keep = codes + 1 * power
            c_add  = codes + 2 * power
            c_sub  = codes + 0 * power
            
            sums = np.concatenate([s_keep, s_add, s_sub])
            codes = np.concatenate([c_keep, c_add, c_sub])
            
        return sums, codes

    # Generate for first half
    sums1, codes1 = generate_sums_and_codes(w1)
    
    if time.time() - start_time > DEADLINE - 10:
        # Fallback: just return a simple heuristic if generation is unexpectedly slow
        # (Unlikely for 3^15)
        return [0]*30

    # Generate for second half
    sums2, codes2 = generate_sums_and_codes(w2)
    
    # Sort sums2 for binary search
    order2 = np.argsort(sums2)
    sums2 = sums2[order2]
    codes2 = codes2[order2]
    
    # Vectorized search for closest non-zero sum
    # We want to minimize |s1 + s2| != 0
    
    # Calculate targets for sums1
    targets = -sums1
    
    # Find insertion points in sums2
    pos = np.searchsorted(sums2, targets)
    
    # Candidate indices: pos and pos-1
    # Clip to valid range [0, len(sums2)-1]
    # We need to be careful with boundaries.
    
    len2 = len(sums2)
    
    # Candidate 1: index = pos (clamped to max len2-1)
    idx1 = np.clip(pos, 0, len2 - 1)
    s2_1 = sums2[idx1]
    diff1 = np.abs(sums1 + s2_1)
    
    # Candidate 2: index = pos - 1 (clamped to min 0)
    idx2 = np.clip(pos - 1, 0, len2 - 1)
    s2_2 = sums2[idx2]
    diff2 = np.abs(sums1 + s2_2)
    
    # Combine differences
    # We want the minimum non-zero difference
    # Create a mask for zero differences
    zero_mask_1 = (diff1 == 0)
    zero_mask_2 = (diff2 == 0)
    
    # If diff is 0, we ignore it. Set to infinity for minimization.
    diff1_valid = np.where(zero_mask_1, np.inf, diff1)
    diff2_valid = np.where(zero_mask_2, np.inf, diff2)
    
    # Find the global minimum
    min_diff = np.minimum(diff1_valid, diff2_valid)
    
    # If all are zero (sum=0 is possible for both halves), we need the next best.
    # But SVP usually has small non-zero vectors.
    # If min_diff is inf, it means all pairs sum to 0? Impossible for distinct weights.
    # Actually, if sums1 and sums2 are such that s1+s2=0 for all s1? No.
    
    # Find the index of the minimum difference in sums1
    # If there are multiple, argmin picks the first.
    min_idx = np.argmin(min_diff)
    
    # Determine which candidate (1 or 2) was better for this min_idx
    # If diff1_valid[min_idx] <= diff2_valid[min_idx], use idx1
    # Else use idx2
    use_idx1 = diff1_valid[min_idx] <= diff2_valid[min_idx]
    
    if use_idx1:
        best_s2_idx = idx1[min_idx]
        best_diff = diff1[min_idx]
    else:
        best_s2_idx = idx2[min_idx]
        best_diff = diff2[min_idx]
        
    # If best_diff is inf, we failed to find a non-zero sum?
    # This would imply all combinations sum to 0, which is impossible.
    # Or we made a mistake in masking.
    # Let's add a safety check.
    if not np.isfinite(best_diff):
        # Fallback to simple local search or return zero
        # Try to find any non-zero sum by scanning a few
        # This case is extremely unlikely.
        return [0]*30
        
    # Retrieve codes
    best_code1 = codes1[min_idx]
    best_code2 = codes2[best_s2_idx]
    
    # Decode coefficients
    coeffs = [0] * 30
    
    # Decode code1 (first n1 coeffs)
    c1 = best_code1
    for j in range(n1):
        digit = c1 % 3
        c1 //= 3
        if digit == 0:
            coeffs[j] = -1
        elif digit == 1:
            coeffs[j] = 0
        else:
            coeffs[j] = 1
            
    # Decode code2 (next n2 coeffs)
    c2 = best_code2
    for j in range(n2):
        digit = c2 % 3
        c2 //= 3
        if digit == 0:
            coeffs[n1 + j] = -1
        elif digit == 1:
            coeffs[n1 + j] = 0
        else:
            coeffs[n1 + j] = 1
            
    return coeffs
