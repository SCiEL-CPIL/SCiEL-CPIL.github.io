import time
import sys
import numpy as np
from bisect import bisect_left

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N
    # Using NumPy for vectorized addition
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # This loop is the bottleneck.
    # Optimization: We can skip even numbers > 2? No, sigma is needed for all.
    # But we can do:
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        # Return a known valid pair if we ran out of time during sieve
        # sigma(11)=12, sigma(6)=12. 11/6 = 1.833, 3/2=1.5. Diff=0.333.
        # We need a valid pair.
        return [11, 6]

    # 2. Group numbers by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = sigma[n]
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() > deadline:
        return [11, 6]

    # 3. Find best pair
    best_diff = float('inf')
    best_a = 11
    best_b = 6
    
    # Process groups
    # Sort groups by size descending? Or just iterate?
    # Iterating is fine.
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        # Sort the numbers in this fiber
        nums.sort()
        
        # Find min |2a - 3b|
        # For each b, find a closest to 1.5b
        for b in nums:
            target_a = 1.5 * b
            # Find position in nums
            pos = bisect_left(nums, target_a)
            
            # Check pos and pos-1
            candidates = []
            if pos < len(nums):
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos-1])
                
            for a in candidates:
                # Calculate exact difference |a/b - 3/2| = |2a - 3b| / (2b)
                # We want to minimize this.
                # Since b > 0, we minimize |2a - 3b| / b
                # Or just compare fractions to avoid float precision issues?
                # The task says "score is the binary64 serialization of that rational".
                # So we compute the float value.
                val = abs(2*a - 3*b) / (2*b)
                if val < best_diff:
                    best_diff = val
                    best_a = a
                    best_b = b
                    if best_diff == 0:
                        # Cannot be 0 as 2a != 3b for integers unless a,b share factors?
                        # 2a = 3b => a=3k, b=2k.
                        # sigma(3k) = sigma(2k)?
                        # If k=1, sigma(3)=4, sigma(2)=3. No.
                        # If found, return immediately? No, 0 is invalid.
                        # So we just track min.
                        pass
                        
        if time.time() > deadline:
            break

    # Ensure types are correct
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
