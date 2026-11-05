import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0

    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def generate_sums_np(weights_np):
        m = len(weights_np)
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_np):
            # Expand sums
            # -1, 0, 1
            # new_sums = [sums - w, sums, sums + w]
            # new_codes = [codes, codes + 3^i, codes + 2*3^i] ?
            # Let's use the base-3 encoding where the i-th digit (0-indexed) corresponds to the i-th weight.
            # If we append the new weight at the end of the processing, its place value is 3^i.
            # Code = c_0 * 3^0 + c_1 * 3^1 + ...
            # c_val in {-1, 0, 1} mapped to {0, 1, 2}?
            # Let's map -1->0, 0->1, 1->2.
            # Then c_val = digit - 1.
            
            new_sums = np.concatenate((sums - w, sums, sums + w))
            
            # Code expansion
            # For each existing code, we generate 3 new codes.
            # If we process weight i, the new digit is at position i.
            # The old code has digits 0..i-1.
            # New code = old_code + digit * 3^i.
            # But we are concatenating, so we need to add the offset to the whole block.
            
            # Block 1: -1 (digit 0). Offset 0.
            # Block 2: 0 (digit 1). Offset 3^i.
            # Block 3: 1 (digit 2). Offset 2 * 3^i.
            
            pow3_i = 3 ** i
            offset1 = 0
            offset2 = pow3_i
            offset3 = 2 * pow3_i
            
            new_codes = np.concatenate((
                codes + offset1, 
                codes + offset2, 
                codes + offset3
            ))
            
            sums = new_sums
            codes = new_codes
            
            if time.time() - start_time > deadline - 10:
                break
                
        return sums, codes

    sums1, codes1 = generate_sums_np(w1)
    sums2, codes2 = generate_sums_np(w2)
    
    if time.time() - start_time > deadline - 5:
        return [1] + [0]*(n-1)

    # Sort sums1 and sums2
    idx1 = np.argsort(sums1)
    sums1_sorted = sums1[idx1]
    codes1_sorted = codes1[idx1]
    
    idx2 = np.argsort(sums2)
    sums2_sorted = sums2[idx2]
    codes2_sorted = codes2[idx2]
    
    # Find best pair
    # For each s1 in sums1_sorted, find s2 in sums2_sorted closest to -s1
    
    # Vectorized searchsorted
    # -sums1_sorted is an array of targets
    targets = -sums1_sorted
    pos = np.searchsorted(sums2_sorted, targets)
    
    # We need to check pos and pos-1
    # Clip pos to valid range for pos-1 and pos
    # pos can be 0 to len(sums2_sorted)
    
    n2 = len(sums2_sorted)
    
    # Candidate 1: sums2_sorted[pos] if pos < n2
    # Candidate 2: sums2_sorted[pos-1] if pos > 0
    
    # To handle boundaries, we can create arrays of indices to check
    # idx_cand1 = pos (if pos < n2 else n2-1)
    # idx_cand2 = pos-1 (if pos > 0 else 0)
    
    idx_cand1 = np.where(pos < n2, pos, n2 - 1)
    idx_cand2 = np.where(pos > 0, pos - 1, 0)
    
    s2_cand1 = sums2_sorted[idx_cand1]
    s2_cand2 = sums2_sorted[idx_cand2]
    
    diff1 = np.abs(sums1_sorted + s2_cand1)
    diff2 = np.abs(sums1_sorted + s2_cand2)
    
    # We want min diff > 0
    # Combine diffs
    # We need to know which candidate gave the min diff to retrieve the code later
    
    # Create a mask for valid non-zero diffs
    # If diff1 == 0, it's invalid. If diff2 == 0, it's invalid.
    
    # Let's find the global minimum non-zero diff
    # We can flatten the two diff arrays?
    # Or just take min of diff1 and diff2, but keep track of source.
    
    min_diff1 = np.min(diff1[diff1 > 0]) if np.any(diff1 > 0) else float('inf')
    min_diff2 = np.min(diff2[diff2 > 0]) if np.any(diff2 > 0) else float('inf')
    
    if min_diff1 < min_diff2:
        best_diff = min_diff1
        # Find index in diff1 where this min occurred
        # Note: diff1[diff1>0] changes indices.
        # Better to use argmin on a modified array
        diff1_valid = np.where(diff1 > 0, diff1, float('inf'))
        best_idx1 = np.argmin(diff1_valid)
        best_s1_idx = best_idx1
        best_s2_idx = idx_cand1[best_idx1]
        best_code1 = codes1_sorted[best_s1_idx]
        best_code2 = codes2_sorted[best_s2_idx]
    else:
        best_diff = min_diff2
        diff2_valid = np.where(diff2 > 0, diff2, float('inf'))
        best_idx2 = np.argmin(diff2_valid)
        best_s1_idx = best_idx2
        best_s2_idx = idx_cand2[best_idx2]
        best_code1 = codes1_sorted[best_s1_idx]
        best_code2 = codes2_sorted[best_s2_idx]
        
    if best_diff == float('inf'):
        return [1] + [0]*(n-1)

    # Decode codes
    def decode_code(code, length):
        coeffs = []
        for i in range(length):
            digit = code % 3
            code //= 3
            # Map 0->-1, 1->0, 2->1
            coeffs.append(digit - 1)
        return coeffs

    c1 = decode_code(best_code1, half)
    c2 = decode_code(best_code2, n - half)
    
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [1] + [0]*(n-1)
        
    return result
