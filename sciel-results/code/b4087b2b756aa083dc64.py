import time
import math
import numpy as np
from collections import defaultdict

def construct():
    # Deadline: 60s limit, keep 5s buffer
    deadline = time.time() + 55
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    # sigma is initialized to 0
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # For each i, add i to sigma[j] for all multiples j of i
    # This is O(N log N) but vectorized in NumPy
    for i in range(1, N + 1):
        if time.time() > deadline - 10:
            # Fallback if sieve takes too long (unlikely with NumPy, but safe)
            # Known valid pair with small error
            return [11, 6]
        
        # Add i to all multiples of i
        sigma[i::i] += i
        
    # Convert to Python list for faster dict lookups? 
    # Actually, iterating over numpy array is okay, but let's stick to it.
    
    # 2. Group numbers by sigma value
    # We only care about sigma values that appear more than once
    # Using a dictionary: key = sigma_value, value = list of n
    
    groups = defaultdict(list)
    
    # Iterating over 10 million elements in Python might take a few seconds
    # We can optimize by checking time periodically
    for n in range(1, N + 1):
        s = sigma[n]
        if s > 0: # sigma is always >= 1
            groups[s].append(n)
            
        if n % 1000000 == 0 and time.time() > deadline - 15:
             # If we are running out of time, we might not finish grouping
             # But grouping is usually faster than the sieve
             break

    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Find best pair
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # We want 2a close to 3b.
    # For each group (fixed sigma), we have a list of numbers L.
    # For each b in L, we want a in L such that a is close to 1.5 * b.
    # Since L is sorted, we can use binary search.
    
    best_err = float('inf')
    best_a = 11
    best_b = 6
    best_num = 1 # numerator of error |2a-3b|
    best_den = 1 # denominator of error 2b
    
    # Iterate through groups
    # To save time, we can prioritize groups with larger sizes or specific properties,
    # but iterating all is safer for finding the global minimum.
    # Number of unique sigma values is roughly N / log(N) or similar, so ~1-2 million groups.
    # Most groups have size 1. We can skip size 1 groups immediately.
    
    # Optimization: Filter groups with size > 1 first?
    # defaultdict(list) already groups them.
    
    for s, nums in groups.items():
        if time.time() > deadline - 2:
            break
            
        k = len(nums)
        if k < 2:
            continue
            
        # Sort the numbers in this group
        # The list is already in increasing order of n because we iterated 1..N
        # So nums is already sorted.
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We can do this efficiently.
        # Note: a and b must be distinct.
        
        # To speed up, we can skip if the minimum possible error for this group 
        # is already worse than best_err? 
        # Min error for group is hard to estimate quickly without checking.
        
        for i in range(k):
            b = nums[i]
            target = 1.5 * b
            
            # Binary search for target in nums
            # We want index where nums[idx] >= target
            lo = 0
            hi = k
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1
            candidates = []
            if lo < k:
                candidates.append(nums[lo])
            if lo > 0:
                candidates.append(nums[lo - 1])
                
            for a in candidates:
                if a == b:
                    continue
                    
                # Calculate error
                # err = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Exact match 3/2 is invalid
                    continue
                den = 2 * b
                err = num / den
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den

    # Final check
    if best_a == best_b or 2 * best_a == 3 * best_b:
        return [11, 6]
        
    return [best_a, best_b]
