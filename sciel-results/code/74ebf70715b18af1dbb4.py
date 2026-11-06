import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1), 1)
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def generate_sums_and_coeffs(weights_subset):
        """
        Generates all possible sums and their corresponding coefficient vectors
        for the given weights.
        Returns:
            sums: numpy array of shape (3^k,)
            coeffs: numpy array of shape (3^k, k) with values in {-1, 0, 1}
        """
        k = len(weights_subset)
        # Start with one sum: 0, and one coeff vector: [0, ..., 0]
        sums = np.array([0], dtype=np.int64)
        coeffs = np.zeros((1, k), dtype=np.int8)
        
        for i, w in enumerate(weights_subset):
            # Expand sums:
            # new_sums = [s - w for s in sums] + [s for s in sums] + [s + w for s in sums]
            # Expand coeffs:
            # Set the i-th column to -1, 0, 1 respectively
            
            # To save memory and time, we construct the new arrays
            # This will triple the size
            
            # We need to be careful with memory allocation.
            # np.concatenate is efficient.
            
            sums_neg = sums - w
            sums_zero = sums
            sums_pos = sums + w
            
            new_sums = np.concatenate([sums_neg, sums_zero, sums_pos])
            
            # For coeffs, we need to copy the existing coeffs and update the i-th column
            # coeffs shape: (N, k)
            # We want (3N, k)
            
            # Copy coeffs 3 times
            c_neg = coeffs.copy()
            c_zero = coeffs.copy()
            c_pos = coeffs.copy()
            
            # Set the i-th column
            c_neg[:, i] = -1
            c_zero[:, i] = 0
            c_pos[:, i] = 1
            
            new_coeffs = np.concatenate([c_neg, c_zero, c_pos], axis=0)
            
            sums = new_sums
            coeffs = new_coeffs
            
            # Check deadline
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded during generation")
                
        return sums, coeffs

    try:
        sums1, coeffs1 = generate_sums_and_coeffs(w1)
        sums2, coeffs2 = generate_sums_and_coeffs(w2)
    except TimeoutError:
        # Fallback to a simpler heuristic if MIM is too slow or memory fails
        # But MIM should be fast enough.
        # Let's try a simple local search fallback
        pass

    # Sort sums2
    sort_indices = np.argsort(sums2)
    sorted_sums2 = sums2[sort_indices]
    # We also need to map back to original indices to get coeffs2
    # Actually, we can just keep the sorted sums and when we find a match,
    # we need the coeffs.
    # It is easier to keep the original coeffs2 and use the index from searchsorted
    # on the sorted sums, then map back to the original index.
    # inverse_indices = np.argsort(sort_indices) is not needed if we just
    # use the index in the sorted array to find the position in the original array?
    # No, searchsorted gives index in sorted_sums2.
    # Let idx_sorted be the index in sorted_sums2.
    # The original index is sort_indices[idx_sorted].
    
    best_abs_N = np.iinfo(np.int64).max
    best_coeffs = None
    
    # Iterate over sums1
    # Vectorized search is better
    # target = -sums1
    # indices = np.searchsorted(sorted_sums2, -sums1)
    
    # We need to check indices and indices-1
    # To avoid out of bounds, we can clip indices
    
    targets = -sums1
    indices = np.searchsorted(sorted_sums2, targets)
    
    # Candidates 1: indices
    # Candidates 2: indices - 1
    
    # Clip indices to valid range [0, len(sorted_sums2)-1]
    idx1 = np.clip(indices, 0, len(sorted_sums2) - 1)
    idx2 = np.clip(indices - 1, 0, len(sorted_sums2) - 1)
    
    # Get sums at these indices
    cand_sums1 = sorted_sums2[idx1]
    cand_sums2 = sorted_sums2[idx2]
    
    # Total sums
    total1 = sums1 + cand_sums1
    total2 = sums1 + cand_sums2
    
    # Absolute values
    abs_total1 = np.abs(total1)
    abs_total2 = np.abs(total2)
    
    # Combine
    # We want the minimum of abs_total1 and abs_total2 for each i
    # But we also need to ensure it's not 0.
    # If min is 0, we need the next smallest.
    
    # Create a mask for non-zero
    # Actually, we can just find the global minimum non-zero.
    
    # Flatten the candidates?
    # Let's just iterate through the vectorized results to find the global min.
    
    # This loop is over 14M elements, which is slow in Python.
    # We need to vectorize the "find min non-zero" logic.
    
    # Combine abs_total1 and abs_total2 into one array of pairs?
    # Or just find min of abs_total1 and min of abs_total2?
    # No, the min for a specific i might be in total1 or total2.
    
    # Let's create a single array of all candidate absolute sums.
    # Shape (2 * 3^15,)
    all_abs = np.concatenate([abs_total1, abs_total2])
    all_orig_idx = np.concatenate([
        np.arange(len(sums1)), 
        np.arange(len(sums1))
    ])
    all_is_second = np.concatenate([
        np.zeros(len(sums1), dtype=bool), 
        np.ones(len(sums1), dtype=bool)
    ])
    # The index into the sorted_sums2 array for the second part
    # For the first part, the index was idx1. For the second, idx2.
    # We need to map back to original sums2 index.
    
    # This is getting complicated to vectorize fully with index tracking.
    # Given 14M elements, a simple loop in Python might take ~5-10 seconds.
    # Is that acceptable? 60s limit.
    # Yes, 10s is fine.
    
    # Let's do a loop.
    
    best_val = np.iinfo(np.int64).max
    best_i1 = -1
    best_i2_sorted = -1
    best_is_second = False
    
    # To speed up, we can filter out zeros first?
    # Most sums are not 0.
    
    # We can use numpy to find the minimum non-zero value in the combined array?
    # Replace 0 with max int, find min, then check if it was 0.
    
    abs_total1_mod = np.where(abs_total1 == 0, np.iinfo(np.int64).max, abs_total1)
    abs_total2_mod = np.where(abs_total2 == 0, np.iinfo(np.int64).max, abs_total2)
    
    # Find global min in these two arrays
    min1 = np.min(abs_total1_mod)
    min2 = np.min(abs_total2_mod)
    
    if min1 < min2:
        # Best is in total1
        best_val = min1
        # Find the index
        idx_in_1 = np.where(abs_total1_mod == min1)[0][0]
        best_i1 = idx_in_1
        best_i2_sorted = idx1[idx_in_1]
        best_is_second = False
    else:
        # Best is in total2
        best_val = min2
        idx_in_2 = np.where(abs_total2_mod == min2)[0][0]
        best_i1 = idx_in_2
        best_i2_sorted = idx2[idx_in_2]
        best_is_second = True
        
    if best_val == np.iinfo(np.int64).max:
        # All sums were 0? Unlikely.
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Retrieve coefficients
    # best_i1 is index in sums1/coeffs1
    # best_i2_sorted is index in sorted_sums2
    # We need the original index in sums2/coeffs2
    # sort_indices maps sorted index -> original index
    
    orig_idx2 = sort_indices[best_i2_sorted]
    
    c1 = coeffs1[best_i1]
    c2 = coeffs2[orig_idx2]
    
    # Combine
    result = np.concatenate([c1, c2]).astype(int).tolist()
    
    # Verify
    # s = sum(result[i] / (i+1) for i in range(30))
    # if s == 0:
    #    # Handle zero case by finding next best
    #    # This is unlikely if we used the mod trick
    
    return result

# EVOLVE-BLOCK-END
