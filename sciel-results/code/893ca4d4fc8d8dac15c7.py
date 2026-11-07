import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    # Deadline is 90s, but we should aim to finish much earlier to be safe.
    # Let's set a hard stop at 75s to allow for cleanup.
    deadline = start_time + 75.0
    
    N = 10000000
    
    # 1. Compute sigma(n) for n in [1, N]
    # Use int32 to save memory and potentially speed up
    sigma = np.zeros(N + 1, dtype=np.int32)
    
    for i in range(1, N + 1):
        if i % 1000000 == 0 and time.time() > deadline - 20.0:
            # If sigma computation is too slow, fall back
            return [23, 15]
        sigma[i::i] += i
        
    if time.time() > deadline - 10.0:
        return [23, 15]

    # 2. Group numbers by sigma value using sorting
    # sigma[0] is 0, we care about 1..N
    sigmas = sigma[1:]
    indices = np.arange(1, N + 1)
    
    # Sort indices by sigma value
    order = np.argsort(sigmas, kind='mergesort') # Stable sort not strictly needed but good
    sorted_sigmas = sigmas[order]
    sorted_indices = indices[order]
    
    # Find group boundaries
    # diff will be non-zero where sigma changes
    diffs = np.diff(sorted_sigmas)
    change_indices = np.where(diffs != 0)[0] + 1
    group_starts = np.concatenate(([0], change_indices))
    group_ends = np.concatenate((change_indices, [len(sorted_sigmas)]))
    
    if time.time() > deadline - 5.0:
        return [23, 15]

    # 3. Search for best pair
    best_error = float('inf')
    best_pair = (23, 15) # Fallback known pair
    
    for start, end in zip(group_starts, group_ends):
        if time.time() > deadline - 2.0:
            break
            
        group_len = end - start
        if group_len < 2:
            continue
            
        # Extract the group of indices (numbers n)
        nums = sorted_indices[start:end]
        # nums is already sorted because sorted_indices is sorted by value, 
        # and within same sigma value, the order of indices is preserved from argsort?
        # No, argsort sorts by key. If keys are equal, the order depends on the sort stability.
        # 'mergesort' is stable, so indices with same sigma will appear in increasing order of their original index.
        # So nums is sorted.
        
        nlen = len(nums)
        
        # For each b in nums, find a in nums such that a/b is close to 1.5
        for i in range(nlen):
            b = int(nums[i])
            target = 1.5 * b
            idx = bisect_left(nums, target)
            
            candidates = set()
            
            if idx < nlen:
                candidates.add(int(nums[idx]))
            if idx > 0:
                candidates.add(int(nums[idx-1]))
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error |a/b - 1.5|
                # |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                
                denom = 2.0 * b
                err = num / denom
                
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
