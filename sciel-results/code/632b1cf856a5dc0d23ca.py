# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    w = np.array([L // (i+1) for i in range(n)], dtype=np.int64)
    
    half = 15
    w_left = w[:half]
    w_right = w[half:]
    
    def gen_sums(weights):
        k = len(weights)
        total = 3 ** k
        vals = np.arange(total, dtype=np.int64)
        sums = np.zeros(total, dtype=np.int64)
        for j in range(k):
            power = 3 ** j
            digits = (vals // power) % 3
            coeffs = digits - 1
            sums += coeffs * weights[j]
        return sums
    
    if time.time() > deadline - 5:
        return [0]*28 + [1, -1]
    
    left_sums = gen_sums(w_left)
    
    if time.time() > deadline - 5:
        return [0]*28 + [1, -1]
    
    right_sums = gen_sums(w_right)
    
    if time.time() > deadline - 5:
        return [0]*28 + [1, -1]
    
    # Sort right sums
    right_order = np.argsort(right_sums)
    right_sorted = right_sums[right_order]
    
    targets = -left_sums
    insert_points = np.searchsorted(right_sorted, targets)
    
    best_num = L // 870  # Current best numerator
    best_li = -1
    best_ri = -1
    
    batch_size = 1000000
    max_int = np.int64(np.iinfo(np.int64).max)
    
    for batch_start in range(0, len(left_sums), batch_size):
        if time.time() > deadline - 10:
            break
        
        batch_end = min(batch_start + batch_size, len(left_sums))
        batch_insert = insert_points[batch_start:batch_end]
        
        valid1 = batch_insert < len(right_sorted)
        cand1_idx = np.where(valid1, batch_insert, 0)
        cand1_vals = right_sorted[cand1_idx]
        total1 = left_sums[batch_start:batch_end] + cand1_vals
        
        valid2 = (batch_insert - 1) >= 0
        cand2_idx = np.where(valid2, batch_insert - 1, 0)
        cand2_vals = right_sorted[cand2_idx]
        total2 = left_sums[batch_start:batch_end] + cand2_vals
        
        abs1 = np.abs(total1)
        abs2 = np.abs(total2)
        
        abs1[~valid1] = max_int
        abs2[~valid2] = max_int
        
        abs1[total1 == 0] = max_int
        abs2[total2 == 0] = max_int
        
        better1 = abs1 <= abs2
        best_abs_batch = np.where(better1, abs1, abs2)
        
        min_idx = int(np.argmin(best_abs_batch))
        if best_abs_batch[min_idx] < best_num:
            best_num = int(best_abs_batch[min_idx])
            best_li = batch_start + min_idx
            if better1[min_idx]:
                best_ri = int(cand1_idx[min_idx])
            else:
                best_ri = int(cand2_idx[min_idx])
            if best_num == 1:
                break
    
    if best_li == -1:
        return [0]*28 + [1, -1]
    
    # Reconstruct coefficients
    def idx_to_coeffs(idx, k):
        powers = 3 ** np.arange(k, dtype=np.int64)
        digits = (idx // powers) % 3
        return digits - 1
    
    left_coeffs = idx_to_coeffs(best_li, half)
    orig_right_idx = int(right_order[best_ri])
    right_coeffs = idx_to_coeffs(orig_right_idx, half)
    
    result = list(left_coeffs.astype(int)) + list(right_coeffs.astype(int))
    
    # Verify
    s_num = sum(int(c) * int(w[i]) for i, c in enumerate(result))
    if s_num == 0:
        return [0]*28 + [1, -1]
    
    return result
# EVOLVE-BLOCK-END
