import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Weights: L // i
    weights = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)
    
    # Split into two halves
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums(w):
        """
        Generate all possible sums for weights w using {-1, 0, 1}.
        Returns a numpy array of sums.
        """
        sums = np.array([0], dtype=np.int64)
        for weight in w:
            # sums + weight, sums, sums - weight
            # This triples the size of the array
            sums = np.concatenate([sums + weight, sums, sums - weight])
        return sums

    # Generate sums for both halves
    # 3^15 = 14,348,907
    # Memory: ~112 MB per array.
    
    if time.time() > deadline - 10:
        return [1] + [0]*29 # Fallback
    
    sums1 = gen_sums(w1)
    sums1.sort()
    
    if time.time() > deadline - 10:
        return [1] + [0]*29 # Fallback
        
    sums2 = gen_sums(w2)
    
    # For each s2 in sums2, we want s1 in sums1 such that |s1 + s2| is min and non-zero.
    # This is equivalent to finding s1 closest to -s2.
    
    # Use searchsorted to find the insertion point of -sums2 in sums1
    # sums1 is sorted.
    # target = -sums2
    targets = -sums2
    
    # searchsorted returns the index where the element would be inserted to maintain order
    # 'left' mode: returns the leftmost insertion point
    idx = np.searchsorted(sums1, targets)
    
    # We need to check the element at idx (successor) and idx-1 (predecessor)
    # Handle boundaries: idx can be 0 or len(sums1)
    
    # Create arrays for the two candidate s1 values
    # Candidate 1: sums1[idx] (if idx < len(sums1))
    # Candidate 2: sums1[idx-1] (if idx > 0)
    
    # To avoid out-of-bounds, we can use clipping or masking.
    # Simpler: create padded arrays or use where clauses.
    
    # Let's use a safe approach:
    # s1_cand1 = sums1[idx] where idx < len, else 0 (and mask out)
    # s1_cand2 = sums1[idx-1] where idx > 0, else 0 (and mask out)
    
    len1 = len(sums1)
    
    # Initialize arrays for candidates
    s1_cand1 = np.zeros(len(sums2), dtype=np.int64)
    s1_cand2 = np.zeros(len(sums2), dtype=np.int64)
    
    valid1 = idx < len1
    valid2 = idx > 0
    
    # Get values
    # np.take is safe with mode='clip' but we want to mask invalid ones anyway.
    # We only care about valid indices.
    
    # For valid1, idx is in [0, len1-1]
    s1_cand1[valid1] = sums1[idx[valid1]]
    
    # For valid2, idx-1 is in [0, len1-1]
    s1_cand2[valid2] = sums1[idx[valid2] - 1]
    
    # Calculate total sums
    total1 = sums2 + s1_cand1
    total2 = sums2 + s1_cand2
    
    # Absolute values
    abs_total1 = np.abs(total1)
    abs_total2 = np.abs(total2)
    
    # Set invalid ones to infinity
    abs_total1[~valid1] = np.iinfo(np.int64).max
    abs_total2[~valid2] = np.iinfo(np.int64).max
    
    # We want non-zero sums.
    # If total is 0, set abs to infinity.
    abs_total1[total1 == 0] = np.iinfo(np.int64).max
    abs_total2[total2 == 0] = np.iinfo(np.int64).max
    
    # Find the global minimum
    # We can combine them into one array and find min, or find min of each and compare.
    min_idx1 = np.argmin(abs_total1)
    min_val1 = abs_total1[min_idx1]
    
    min_idx2 = np.argmin(abs_total2)
    min_val2 = abs_total2[min_idx2]
    
    if min_val1 <= min_val2:
        best_s2 = sums2[min_idx1]
        best_s1 = s1_cand1[min_idx1]
        best_val = min_val1
    else:
        best_s2 = sums2[min_idx2]
        best_s1 = s1_cand2[min_idx2]
        best_val = min_val2
        
    if best_val == np.iinfo(np.int64).max:
        # No non-zero sum found? Impossible for N=30.
        return [1] + [0]*29
        
    # Reconstruct coefficients
    def find_coeffs(weights_subset, target_sum):
        n = len(weights_subset)
        coeffs = [0] * n
        
        # Precompute suffix sums for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights_subset[i]
            
        def dfs(pos, current_sum):
            if pos == n:
                return current_sum == target_sum
            
            w = weights_subset[pos]
            
            # Pruning
            # Try 1
            new_sum = current_sum + w
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = 1
                if dfs(pos + 1, new_sum):
                    return True
            
            # Try 0
            new_sum = current_sum
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = 0
                if dfs(pos + 1, new_sum):
                    return True
            
            # Try -1
            new_sum = current_sum - w
            if abs(target_sum - new_sum) <= suffix_sums[pos + 1]:
                coeffs[pos] = -1
                if dfs(pos + 1, new_sum):
                    return True
            
            coeffs[pos] = 0
            return False
            
        if dfs(0, 0):
            return coeffs
        return None

    c1 = find_coeffs(w1, best_s1)
    c2 = find_coeffs(w2, best_s2)
    
    if c1 is None or c2 is None:
        # Fallback
        return [1] + [0]*29
        
    return list(c1) + list(c2)
