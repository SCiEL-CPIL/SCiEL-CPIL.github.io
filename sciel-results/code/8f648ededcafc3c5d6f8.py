"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Calculate weights W_i = L // i
    weights = [L // i for i in range(1, 31)]
    
    # 3. Split weights into two halves (15 each)
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums_with_indices(ws):
        """
        Generate all possible signed sums for the given weights.
        Returns (sums_array, indices_array) where indices_array[i] is the base-3 index
        that produced sums_array[i].
        """
        sums = np.array([0], dtype=np.int64)
        indices = np.array([0], dtype=np.int32)
        for w in ws:
            n = len(sums)
            new_sums = np.empty(3 * n, dtype=np.int64)
            new_indices = np.empty(3 * n, dtype=np.int32)
            
            new_sums[0:n] = sums - w
            new_indices[0:n] = indices * 3 + 0
            
            new_sums[n:2*n] = sums
            new_indices[n:2*n] = indices * 3 + 1
            
            new_sums[2*n:3*n] = sums + w
            new_indices[2*n:3*n] = indices * 3 + 2
            
            sums = new_sums
            indices = new_indices
        return sums, indices
    
    # Generate sums and indices for both halves
    sums1, idx1 = generate_sums_with_indices(half1_weights)
    sums2, idx2 = generate_sums_with_indices(half2_weights)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
    
    # 4. Sort sums2 and keep track of original indices
    sort_order = np.argsort(sums2)
    sorted_sums2 = sums2[sort_order]
    sorted_idx2 = idx2[sort_order]
    
    # 5. For each sum in half1, find the best matching sum in half2
    #    We want sums1[i] + sums2[j] to be as close to 0 as possible (but nonzero)
    
    # Use searchsorted to find insertion points
    neg_sums1 = -sums1
    insert_pos = np.searchsorted(sorted_sums2, neg_sums1, side='left')
    
    # Check candidates at insert_pos and insert_pos-1
    n_sums2 = len(sorted_sums2)
    
    # Candidate 1: at insert_pos (clipped)
    pos1 = np.clip(insert_pos, 0, n_sums2 - 1)
    # Candidate 2: at insert_pos - 1 (clipped)
    pos2 = np.clip(insert_pos - 1, 0, n_sums2 - 1)
    
    cand1 = sorted_sums2[pos1]
    cand2 = sorted_sums2[pos2]
    
    total1 = sums1 + cand1
    total2 = sums1 + cand2
    
    abs_total1 = np.where(total1 != 0, np.abs(total1), np.iinfo(np.int64).max)
    abs_total2 = np.where(total2 != 0, np.abs(total2), np.iinfo(np.int64).max)
    
    # Combine: for each i, take the better of the two candidates
    best_abs = np.minimum(abs_total1, abs_total2)
    best_pos = np.where(abs_total1 <= abs_total2, pos1, pos2)
    
    # Find the global minimum
    min_abs = np.min(best_abs)
    
    if min_abs == np.iinfo(np.int64).max:
        return [0] * 28 + [1, -1]
    
    k1_idx = np.argmin(best_abs)
    pos_in_sorted = best_pos[k1_idx]
    orig_pos = sort_order[pos_in_sorted]
    idx_in_sums2 = orig_pos
    idx_in_sums1 = k1_idx
    
    # 6. Decode indices to coefficients
    def decode_index(k, n_weights):
        coeffs = []
        for _ in range(n_weights):
            digit = k % 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
            k //= 3
        coeffs.reverse()
        return coeffs

    coeffs1 = decode_index(idx1[idx_in_sums1], len(half1_weights))
    coeffs2 = decode_index(idx2[idx_in_sums2], len(half2_weights))
    
    result = coeffs1 + coeffs2
    
    # Verify: compute the actual sum
    actual_sum = 0
    for i, c in enumerate(result):
        if c != 0:
            actual_sum += c * (L // (i + 1))
    
    if actual_sum == 0:
        # Fallback: try a simple nonzero combination
        return [0] * 28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
