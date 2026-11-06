import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5 second safety margin
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums and indices for a given weight list
    # Uses recursive doubling which is faster than decoding base-3 indices
    def gen_sums(w):
        sums = [0]
        idxs = [0]
        for i, weight in enumerate(w):
            pow3 = 3 ** i
            new_sums = []
            new_idxs = []
            # Optimization: list comprehension might be faster, but explicit loop is clearer
            # and allows interleaving logic if needed. 
            # Let's try to optimize the inner loop.
            for s, idx in zip(sums, idxs):
                new_sums.append(s - weight)
                new_idxs.append(idx)
                
                new_sums.append(s)
                new_idxs.append(idx + pow3)
                
                new_sums.append(s + weight)
                new_idxs.append(idx + 2 * pow3)
            sums = new_sums
            idxs = new_idxs
        return sums, idxs

    # Generate first half
    sums1, idxs1 = gen_sums(w1)
    
    # Prepare for binary search
    # Convert to numpy arrays for fast sorting and search
    sums1_arr = np.array(sums1, dtype=np.int64)
    idxs1_arr = np.array(idxs1, dtype=np.int64)
    
    sorted_order = np.argsort(sums1_arr)
    sorted_sums1_vals = sums1_arr[sorted_order]
    sorted_idxs1 = idxs1_arr[sorted_order]
    
    best_abs = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    # Generate second half
    sums2, idxs2 = gen_sums(w2)
    
    # Iterate over second half sums
    for s2, idx2 in zip(sums2, idxs2):
        target = -s2
        # Find closest in sorted_sums1_vals
        pos = np.searchsorted(sorted_sums1_vals, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sorted_sums1_vals):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for p in candidates:
            s1_val = sorted_sums1_vals[p]
            idx1 = sorted_idxs1[p]
            total_sum = s1_val + s2
            if total_sum == 0:
                continue
            abs_sum = abs(total_sum)
            if abs_sum < best_abs:
                best_abs = abs_sum
                best_idx1 = idx1
                best_idx2 = idx2
                
        if time.time() > deadline:
            break

    if best_idx1 == -1:
        # Fallback
        return [0]*28 + [1, -1]

    # Decode indices to coefficients
    def decode(idx, k):
        coeffs = []
        rem = idx
        for j in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs

    c1 = decode(best_idx1, half)
    c2 = decode(best_idx2, n - half)
    
    return c1 + c2
