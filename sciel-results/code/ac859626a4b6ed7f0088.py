import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin
    
    MAX_N = 10000000
    
    # 1. Compute sigma(n) for all n in [1, MAX_N]
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This loop is O(N log N) but vectorized
    for i in range(1, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        sigma[i::i] += i
        
    if time.time() - start_time > DEADLINE:
        # Fallback: return a known valid pair if we ran out of time during sieve
        # sigma(11) = 1 + 11 = 12
        # sigma(6) = 1 + 2 + 3 + 6 = 12
        # |11/6 - 1.5| = |1.833 - 1.5| = 0.333
        # This is a valid pair.
        return [11, 6]

    # 2. Group indices by sigma value
    # Using a dictionary of lists
    groups = {}
    # It's faster to iterate in Python if we just append, but 
    # np.unique on the sigma array (excluding 0) might be faster to get unique keys,
    # but we need the mapping.
    
    # Let's try building the dictionary.
    # To speed up, we can skip sigma[0] which is 0.
    # We can use np.nonzero or just iterate.
    # Iterating 10^7 times in Python is ~0.5-1s.
    
    for n in range(1, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    # 3. Search for best pair
    best_err = float('inf')
    best_pair = (11, 6)
    
    # We want to minimize |a/b - 1.5|
    # This is equivalent to minimizing |2a - 3b|
    
    for s, nums in groups.items():
        if time.time() - start_time > DEADLINE:
            break
            
        if len(nums) < 2:
            continue
            
        # nums is already sorted because we iterated n in increasing order
        n_len = len(nums)
        
        # For each b in nums, find a in nums such that a is close to 1.5 * b
        # We can use bisect
        for b in nums:
            target = 1.5 * b
            # Find insertion point for target in nums
            # We want a close to target
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            # a must be != b
            
            candidates = []
            if idx < n_len:
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error
                # err = abs(a/b - 1.5)
                # To avoid floating point issues and division, use |2a - 3b| / (2b)
                # Since we want to minimize this, and b is positive, 
                # minimizing |2a - 3b| / b is not quite right if b varies.
                # The objective is |a/b - 1.5|.
                # Let's just compute the float error.
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
        # Optional: early termination if best_err is extremely small
        if best_err < 1e-10:
            break

    a, b = best_pair
    return [int(a), int(b)]
