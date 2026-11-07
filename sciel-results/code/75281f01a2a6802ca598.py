import numpy as np
import math

def construct():
    N = 10000000
    
    # Step 1: Compute sigma(n) using NumPy sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    # This is O(N log N) but vectorized
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Step 2: Sort numbers by their sigma value
    # We want to group numbers with the same sigma
    # argsort gives indices that would sort the array
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    # Step 3: Iterate through groups and find best pair
    best_error_num = None
    best_error_den = None
    best_a = None
    best_b = None
    
    # Iterate to find groups
    # We can use np.diff to find where sigmas change
    # But iterating in Python is safer for the logic
    
    # To speed up, we can skip groups of size 1
    # We identify start and end of each group
    
    # Let's just loop through the sorted array
    # This might be slow if done in pure Python for 10M items.
    # We can use a while loop with indices.
    
    i = 0
    n_items = len(sorted_indices)
    
    while i < n_items:
        current_sigma = sorted_sigmas[i]
        j = i + 1
        # Find end of group
        while j < n_items and sorted_sigmas[j] == current_sigma:
            j += 1
            
        group_size = j - i
        if group_size >= 2:
            # Extract the numbers in this group
            # sorted_indices[i:j] are the numbers
            nums = sorted_indices[i:j]
            # Sort them by value
            nums.sort()
            
            nlen = len(nums)
            # Find best pair in nums
            # Same logic as before
            for k in range(nlen):
                b = nums[k]
                target = 1.5 * b
                # Binary search in nums
                # Use np.searchsorted for speed?
                # np.searchsorted is C-implemented
                idx = np.searchsorted(nums, target)
                
                # Check idx and idx-1
                for idx_cand in (idx - 1, idx):
                    if 0 <= idx_cand < nlen:
                        a = nums[idx_cand]
                        if a == b:
                            continue
                        if 2 * a == 3 * b:
                            continue
                        err_num = abs(2 * a - 3 * b)
                        if err_num == 0:
                            continue
                        err_den = 2 * b
                        
                        if best_error_num is None:
                            best_error_num = err_num
                            best_error_den = err_den
                            best_a = a
                            best_b = b
                        else:
                            if err_num * best_error_den < best_error_num * err_den:
                                best_error_num = err_num
                                best_error_den = err_den
                                best_a = a
                                best_b = b
                                
        i = j
        
    if best_a is None:
        return [11, 6]
        
    return [int(best_a), int(best_b)]
