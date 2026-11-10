import time
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000
    
    # 1. Precompute sigma(n) for all n up to MAX_N
    # Using the linear sieve approach from the current program for speed,
    # but verified to be correct.
    # sigma(n) = product over p^k || n of (1 + p + ... + p^k)
    
    sigma = [0] * (MAX_N + 1)
    spow = [0] * (MAX_N + 1)
    sval = [0] * (MAX_N + 1)
    is_comp = [False] * (MAX_N + 1)
    primes = []
    
    sigma[1] = 1
    spow[1] = 1
    sval[1] = 1
    
    for i in range(2, MAX_N + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            sval[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            j = i * p
            if j > MAX_N:
                break
            is_comp[j] = True
            if i % p == 0:
                spow[j] = spow[i] * p
                sval[j] = sval[i] + spow[j]
                sigma[j] = (sigma[i] // sval[i]) * sval[j]
                break
            else:
                spow[j] = p
                sval[j] = 1 + p
                sigma[j] = sigma[i] * (1 + p)
                
    if time.time() > deadline - 10:
        return [11, 6]
        
    # 2. Group numbers by their sigma value
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for pairs (a, b) in the same fiber minimizing |a/b - 3/2|
    # Error = |2a - 3b| / (2b)
    # We want small |2a - 3b| and large b.
    
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # Iterate through fibers with size > 1
    # To improve efficiency, we can sort the numbers in each fiber and use binary search.
    # We prioritize fibers with larger numbers? Not necessarily, but larger b gives smaller error for same |2a-3b|.
    # However, we must check all to be sure.
    
    # Optimization: 
    # Instead of checking every b in every fiber, we can just check the closest a to 1.5b.
    # If we find a very small error, we might stop early?
    # The theoretical min error is 0.
    
    # Let's sort the list of fibers by the maximum element in descending order?
    # This might help find large b early, allowing us to prune some searches?
    # Actually, binary search is O(log N) per b.
    # Total cost is sum over fibers of |F| log |F|.
    # This is bounded by N log N, which is fine.
    
    # Let's just iterate through all fibers.
    for s_val, nums in fibers.items():
        if time.time() > deadline - 1:
            break
            
        if len(nums) < 2:
            continue
            
        # Sort the list of numbers in this fiber
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We can use bisect
        
        for b in nums:
            if time.time() > deadline - 1:
                break
                
            target_a = 1.5 * b
            # Find insertion point
            pos = bisect.bisect_left(nums, target_a)
            
            # Check neighbors: pos-1, pos
            # pos is where target_a would be inserted to keep list sorted.
            # So nums[pos-1] < target_a <= nums[pos] (if pos < len)
            # Candidates: pos-1, pos
            
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue # a and b must be distinct? 
                    # The problem says "Approximate a ratio... sigma(a)=sigma(b)".
                    # It doesn't explicitly say a != b.
                    # If a == b, a/b = 1, |1 - 1.5| = 0.5.
                    # But usually these tasks imply distinct or just any pair.
                    # If a=b is allowed, and we find a pair with a=b, error is 0.5.
                    # But we can likely do better.
                    # Let's assume a and b can be equal, but it's unlikely to be optimal.
                    
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
                    # If we hit 0, we can't do better.
                    if err == 0.0:
                        return [int(best_a), int(best_b)]
                        
    return [int(best_a), int(best_b)]
