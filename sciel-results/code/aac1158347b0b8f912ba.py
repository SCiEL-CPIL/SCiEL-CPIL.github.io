import time
import numpy as np
from collections import defaultdict
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N]
    # Using NumPy for speed
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for divisor sum
    # This loop is O(N log N) but vectorized
    for d in range(1, N + 1):
        sigma[d::d] += d
        
        # Check time occasionally
        if d % 1_000_000 == 0:
            if time.time() > deadline - 20:
                # If we are running out of time, we might need to stop and process what we have?
                # But sigma computation needs to be complete for valid pairs.
                # If we stop early, sigma is wrong for large n.
                # So we must finish the sieve or reduce N.
                # 10^7 sieve usually takes < 5-10s in NumPy.
                pass

    # 2. Group numbers by sigma value
    # We only care about groups with at least 2 elements.
    # Iterating 10M elements in Python is ~1-2s.
    groups = defaultdict(list)
    
    # Convert to list for faster iteration? Or just iterate range.
    # sigma is a numpy array.
    # We can use np.unique? No, we need the indices.
    # Iterating is safer for memory control if we filter small groups?
    # Actually, just building the dict is fine.
    
    for n in range(1, N + 1):
        s = sigma[n]
        if s > 0:
            groups[s].append(n)
            
    if time.time() > deadline - 10:
        # Fallback to seed if grouping takes too long? Unlikely.
        return [11, 6]

    # 3. Search for best pair
    best_a = 11
    best_b = 6
    # Error for 11/6: |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3
    # Numerator: |2*11 - 3*6| = |22 - 18| = 4
    # Denominator: 2*6 = 12
    # Error = 4/12 = 1/3
    best_num = 4
    best_den = 12

    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        # nums is already sorted because we iterated n from 1 to N
        # We want to minimize |2a - 3b| / (2b)
        # For each a, find b close to 2a/3
        
        for a in nums:
            # We want b such that b ~ 2a/3
            target = (2.0 * a) / 3.0
            
            # Find insertion point in nums
            # bisect_left works on lists
            idx = bisect_left(nums, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for b in candidates:
                if b == a:
                    continue # a != b is implied by sigma(a)=sigma(b) but a=b gives error 1.5, not min. 
                             # Actually if a=b, |1 - 1.5| = 0.5. 
                             # We want small error.
                             
                # Calculate error numerator and denominator
                # E = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                den = 2 * b
                
                # Compare with best
                # diff / den < best_num / best_den
                # diff * best_den < best_num * den
                if diff * best_den < best_num * den:
                    best_num = diff
                    best_den = den
                    best_a = a
                    best_b = b
                    
        # Check time
        if time.time() > deadline - 5:
            break

    # Ensure types are Python int
    return [int(best_a), int(best_b)]
