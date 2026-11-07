import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    N = 10000000
    
    # 1. Compute sigma(n) using a fast sieve
    # Using a standard sieve approach.
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Optimization: 
    # The loop `for i in range(1, N+1): sigma[i::i] += i` 
    # is O(N log N).
    # For large i, the slice is small.
    # We can split the loop?
    # No, NumPy handles it well.
    
    # Time check during sigma computation
    # This is the bottleneck.
    # If it takes too long, we fallback.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            # Fallback to a known small pair if sigma computation is too slow
            return [23, 15]
        sigma[i::i] += i
        
    if time.time() > deadline - 15.0:
        return [23, 15]

    # 2. Group numbers by sigma value
    # We need to map sigma_value -> list of n
    # Using a dictionary.
    groups = {}
    
    # Converting to list for faster indexing in Python loop?
    # sigma is np.array. Accessing sigma[n] is fast enough.
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
        
    if time.time() > deadline - 5.0:
        return [23, 15]

    # 3. Search for best pair
    best_error = float('inf')
    best_pair = (23, 15) # Fallback known pair
    
    # Iterate through groups
    for s, nums in groups.items():
        if time.time() > deadline - 2.0:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        nlen = len(nums)
        
        # For each b in nums, find a in nums such that a/b is close to 1.5
        # a approx 1.5 * b
        # We can iterate b.
        # To save time, we can iterate only a subset if the group is huge, 
        # but most groups are small.
        
        for b in nums:
            # Target a is 1.5 * b
            target = 1.5 * b
            idx = bisect_left(nums, target)
            
            candidates = set()
            
            if idx < nlen:
                candidates.add(nums[idx])
            if idx > 0:
                candidates.add(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error |a/b - 1.5|
                # |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Exact match, but problem says "raw value of exactly 0 cannot be scored and counts as invalid"
                    # So we must avoid a/b == 1.5 exactly.
                    # This means 2a == 3b. Since a,b integers, b must be even, a=3b/2.
                    # If we find such a pair, we must skip it or find another.
                    # The problem implies we need a non-zero error.
                    continue
                
                denom = 2.0 * b
                err = num / denom
                
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
