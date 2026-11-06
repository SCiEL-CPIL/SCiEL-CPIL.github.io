import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Weights for i=1..30 are L/i
    # Indices 0..29 correspond to denominators 1..30
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_and_codes(weights_half):
        # Start with sum 0 and code 0
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for w in weights_half:
            # New sums
            s_minus = sums - w
            s_zero = sums
            s_plus = sums + w
            
            # New codes
            # Digit 0 -> -1, Digit 1 -> 0, Digit 2 -> 1
            # We append the new digit to the code.
            # If we treat the code as a base-3 number, and we process weights from 0 to k,
            # and we do code * 3 + digit, then the first processed weight becomes the most significant digit.
            c_minus = codes * 3      # digit 0 (-1)
            c_zero = codes * 3 + 1   # digit 1 (0)
            c_plus = codes * 3 + 2   # digit 2 (1)
            
            sums = np.concatenate([s_minus, s_zero, s_plus])
            codes = np.concatenate([c_minus, c_zero, c_plus])
            
        return sums, codes

    # Generate for first half
    sums1, codes1 = generate_sums_and_codes(w1)
    
    # Sort first half sums and align codes
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    codes1_sorted = codes1[order1]
    
    # Generate for second half
    sums2, codes2 = generate_sums_and_codes(w2)
    
    len_s1 = len(sums1_sorted)
    
    # We want s1 + s2 close to 0 => s1 close to -s2
    targets = -sums2
    
    # Find insertion points
    # idx_l is the first index in sums1_sorted >= target
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    
    # Candidate 1: idx_l (successor or equal)
    # Candidate 2: idx_l - 1 (predecessor)
    
    # Handle bounds for idx_l
    idx_succ = np.clip(idx_l, 0, len_s1 - 1)
    valid_succ = (idx_l < len_s1) # If idx_l == len_s1, the element at idx_l-1 is the max, but we clipped to len_s1-1. 
                                  # Actually, if idx_l == len_s1, there is no successor. 
                                  # Let's just mask invalid indices.
    
    # Let's be precise.
    # idx_l in [0, len_s1]
    # If idx_l == len_s1, no element >= target. The closest is at len_s1 - 1.
    # If idx_l == 0, no element < target. The closest is at 0.
    
    # Successor candidates
    valid_succ_mask = idx_l < len_s1
    idx_succ_clipped = np.clip(idx_l, 0, len_s1 - 1)
    
    s1_succ = sums1_sorted[idx_succ_clipped]
    c1_succ = codes1_sorted[idx_succ_clipped]
    
    total_succ = sums2 + s1_succ
    abs_total_succ = np.abs(total_succ)
    abs_total_succ[~valid_succ_mask] = np.iinfo(np.int64).max
    
    # Predecessor candidates
    valid_pred_mask = idx_l > 0
    idx_pred_clipped = np.clip(idx_l - 1, 0, len_s1 - 1)
    
    s1_pred = sums1_sorted[idx_pred_clipped]
    c1_pred = codes1_sorted[idx_pred_clipped]
    
    total_pred = sums2 + s1_pred
    abs_total_pred = np.abs(total_pred)
    abs_total_pred[~valid_pred_mask] = np.iinfo(np.int64).max
    
    # Combine and find min
    # We need the smallest non-zero absolute sum.
    
    # Mask out zeros
    abs_total_succ_safe = abs_total_succ.copy()
    abs_total_succ_safe[total_succ == 0] = np.iinfo(np.int64).max
    
    abs_total_pred_safe = abs_total_pred.copy()
    abs_total_pred_safe[total_pred == 0] = np.iinfo(np.int64).max
    
    min_succ = np.min(abs_total_succ_safe)
    min_pred = np.min(abs_total_pred_safe)
    
    if min_pred < min_succ:
        best_abs = min_pred
        idx_best_2 = np.argmin(abs_total_pred_safe)
        best_code1 = c1_pred[idx_best_2]
        best_code2 = codes2[idx_best_2]
    else:
        best_abs = min_succ
        idx_best_2 = np.argmin(abs_total_succ_safe)
        best_code1 = c1_succ[idx_best_2]
        best_code2 = codes2[idx_best_2]
        
    # If all sums were 0 (impossible for 30 terms with this structure? Sum of 1/i is irrational? No, rational. 
    # But is it possible that ALL pairs sum to 0? No.
    # If best_abs is still max, something went wrong, but we can return zeros (invalid) or handle it.
    # Given the density, a non-zero sum exists.
    
    def decode_code(code, num_weights):
        coeffs = [0] * num_weights
        c = int(code)
        for i in range(num_weights - 1, -1, -1):
            digit = c % 3
            c //= 3
            coeffs[i] = digit - 1
        return coeffs

    c1 = decode_code(best_code1, len(w1))
    c2 = decode_code(best_code2, len(w2))
    
    final_coeffs = c1 + c2
    
    # Optional: Verification
    # val = sum(final_coeffs[i] / (i+1) for i in range(n))
    # if val == 0:
    #    # Fallback: find next best
    #    pass
    
    return final_coeffs
