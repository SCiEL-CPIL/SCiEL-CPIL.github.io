import numpy as np
import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 30
    L = reduce(math.lcm, range(1, n + 1))
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    mid = n // 2  # 15
    
    def generate_sums_and_indices_np(weights_subset):
        """
        Generates all possible sums and their base-3 indices for a subset of weights.
        Returns two numpy arrays: sums, indices.
        """
        sums = np.array([0], dtype=np.int64)
        indices = np.array([0], dtype=np.int64)
        power = np.array(1, dtype=np.int64)
        
        for w in weights_subset:
            # For each existing sum, create 3 new sums
            # s - w, s, s + w
            # idx, idx + power, idx + 2*power
            
            # We need to repeat each element 3 times
            # sums_new = np.concatenate([sums - w, sums, sums + w])
            # indices_new = np.concatenate([indices, indices + power, indices + 2*power])
            
            # This creates 3 copies of the current arrays
            # This is memory efficient if we do it in place? No, concatenate is fine.
            
            sums = np.concatenate([sums - w, sums, sums + w])
            indices = np.concatenate([indices, indices + power, indices + 2*power])
            
            power *= 3
            
            if time.time() > deadline:
                return None, None
                
        return sums, indices

    w1 = weights[:mid]
    sums1, indices1 = generate_sums_and_indices_np(w1)
    if sums1 is None:
        return [0] * 28 + [1, -1]
        
    w2 = weights[mid:]
    sums2, indices2 = generate_sums_and_indices_np(w2)
    if sums2 is None:
        return [0] * 28 + [1, -1]

    # Sort sums2 and permute indices2 accordingly
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    indices2_sorted = indices2[sort_idx]
    
    best_val = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2 = -1
    
    # For each s1 in sums1, find s2 in sums2_sorted closest to -s1
    # We can use searchsorted
    
    # We want to avoid iterating 14 million times in Python.
    # Can we vectorize the search?
    # searchsorted returns the index where the value would be inserted.
    # We can do this for all sums1 at once!
    
    targets = -sums1
    positions = np.searchsorted(sums2_sorted, targets)
    
    # We need to check positions and positions-1
    # Clamp positions to valid range
    pos1 = positions
    pos0 = positions - 1
    
    # Handle boundaries
    pos1 = np.clip(pos1, 0, len(sums2_sorted) - 1)
    pos0 = np.clip(pos0, 0, len(sums2_sorted) - 1)
    
    # Get candidate s2 values
    s2_candidates_1 = sums2_sorted[pos1]
    s2_candidates_0 = sums2_sorted[pos0]
    
    # Calculate total sums
    total_sums_1 = sums1 + s2_candidates_1
    total_sums_0 = sums1 + s2_candidates_0
    
    # We want min |total_sum| where total_sum != 0
    
    # Create masks for non-zero
    mask1 = total_sums_1 != 0
    mask0 = total_sums_0 != 0
    
    # Calculate absolute values, setting zero sums to infinity
    abs_total_1 = np.abs(total_sums_1)
    abs_total_0 = np.abs(total_sums_0)
    
    # Mask out zeros
    abs_total_1[~mask1] = np.iinfo(np.int64).max
    abs_total_0[~mask0] = np.iinfo(np.int64).max
    
    # Find the minimum for each s1
    min_abs_1 = abs_total_1
    min_abs_0 = abs_total_0
    
    # We need to pick the better of the two candidates for each s1
    # If min_abs_0 < min_abs_1, pick candidate 0, else candidate 1
    
    use_candidate_0 = min_abs_0 < min_abs_1
    
    # We need the global minimum across all s1
    
    # Combine the absolute values into a single array to find the global min
    # But we also need to know which s1 and which candidate was chosen
    
    # Let's create an array of the best absolute value for each s1
    best_abs_for_s1 = np.where(use_candidate_0, min_abs_0, min_abs_1)
    
    # Find the index of the global minimum
    global_min_idx = np.argmin(best_abs_for_s1)
    global_min_val = best_abs_for_s1[global_min_idx]
    
    if global_min_val == np.iinfo(np.int64).max:
        # No non-zero sum found? Should not happen.
        return [0] * 28 + [1, -1]
        
    # Determine which candidate was used for this s1
    if use_candidate_0[global_min_idx]:
        s2_idx = pos0[global_min_idx]
    else:
        s2_idx = pos1[global_min_idx]
        
    best_idx1 = indices1[global_min_idx]
    best_idx2 = indices2_sorted[s2_idx]
    
    # Reconstruct coefficients
    coeffs = [0] * n
    
    # Decode idx1 for first half
    temp_idx = int(best_idx1)
    for i in range(mid):
        digit = temp_idx % 3
        coeffs[i] = digit - 1
        temp_idx //= 3
        
    # Decode idx2 for second half
    temp_idx = int(best_idx2)
    for i in range(mid, n):
        digit = temp_idx % 3
        coeffs[i] = digit - 1
        temp_idx //= 3
        
    return coeffs
