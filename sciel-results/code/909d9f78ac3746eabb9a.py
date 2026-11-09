import math
import time
import bisect
import numpy as np

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    # sigma(n) = sum of divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes style for sum of divisors
    # for d in 1..N: add d to all multiples of d
    for d in range(1, MAX_N + 1):
        if time.time() > deadline - 10:
            # If we run out of time here, we might have incomplete sigma.
            # But the sieve is the most critical part. 
            # Let's just hope it finishes. It's O(N log N).
            # 10^7 * log(10^7) ~ 1.6 * 10^8 ops. Numpy vector ops are fast.
            pass
        sigma[d::d] += d
        
    # 2. Group numbers by their sigma value
    groups = {}
    # Iterate through sigma array
    # To speed up, we can use a list of lists and index by sigma?
    # Max sigma value is roughly N * 1.5? 
    # Actually, sigma(n) < n * (1 + 1/2 + ... + 1/n) ~ n log n.
    # For n=10^7, sigma(n) can be up to ~ 50 million.
    # A dictionary is safer.
    
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() > deadline - 5:
        # Fallback: return a known valid pair if we ran out of time
        # We need a valid pair. 
        # Let's just return a dummy valid pair found early or a hardcoded one.
        # But we haven't found one yet in this code path if we broke early.
        # We should not break early in the sieve if we want a good score.
        # The sieve is fast enough.
        pass

    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # 3. Search for best pair in each group
    for s, nums in groups.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
            
        # Sort the list to allow binary search
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5*b
        # We can iterate b and use bisect on the same sorted list
        for b in nums:
            target_a = 1.5 * b
            # Find position where target_a would be inserted
            idx = bisect.bisect_left(nums, target_a)
            
            # Check candidates around idx
            # idx, idx-1, idx+1, idx-2, idx+2
            # We want to minimize |a/b - 1.5|
            # which is equivalent to minimizing |2a - 3b| / (2b)
            # Since b is fixed for this iteration, minimizing |2a - 3b| is sufficient locally,
            # but we compare global errors later.
            
            candidates = []
            for offset in [-2, -1, 0, 1, 2]:
                cand_idx = idx + offset
                if 0 <= cand_idx < len(nums):
                    a = nums[cand_idx]
                    candidates.append(a)
            
            for a in candidates:
                if a == b:
                    # Error is |1 - 1.5| = 0.5. Valid, but likely not minimal.
                    # We can still consider it if no better found.
                    pass
                
                # Calculate exact error
                # err = |a/b - 1.5|
                # To avoid floating point issues in comparison, use fractions or high precision?
                # The task says "score is the binary64 serialization of that rational".
                # So we can use float for comparison, but be careful.
                # err = abs(a / b - 1.5)
                
                # Check if error is 0
                # a/b = 1.5 => 2a = 3b
                if 2 * a == 3 * b:
                    # Error is 0. Invalid.
                    continue
                
                err = abs(a / b - 1.5)
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    # Ensure we return Python ints
    return [int(best_a), int(best_b)]
