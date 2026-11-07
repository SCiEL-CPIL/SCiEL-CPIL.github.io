"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 10000000
    
    # 1. Sieve for sigma: sigma[n] = sum of divisors of n
    # Using NumPy for vectorized updates
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        sigma[d::d] += d
        
        # Check deadline periodically to avoid hard kill
        if d % 100000 == 0:
            if time.time() - t0 > DEADLINE:
                # Fallback if sieve is unexpectedly slow (should not happen with NumPy)
                return [11, 6]
                
    if time.time() - t0 > DEADLINE:
        return [11, 6]
        
    # 2. Group by sigma value
    # We use a dictionary mapping sigma value to list of n
    groups = defaultdict(list)
    # Converting to a list of tuples might be faster for iteration than numpy array indexing in Python loop?
    # Actually, iterating over numpy array is fast if we don't do complex logic.
    # But we need to append to lists.
    # Let's iterate over range(1, N+1)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() - t0 > DEADLINE:
        return [11, 6]
        
    # 3. Find best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let num = |2a - 3b|, den = 2b
    # We compare num/den against best_num/best_den
    # num * best_den < best_num * den
    
    best_num = 4  # |2*11 - 3*6| = |22 - 18| = 4
    best_den = 12 # 2*6
    best_a = 11
    best_b = 6
    
    # Iterate through groups
    for s, nums in groups.items():
        if time.time() - t0 > DEADLINE:
            break
            
        if len(nums) < 2:
            continue
            
        # nums is already sorted because we appended n in increasing order
        # For each b in nums, find a in nums closest to 1.5 * b
        for b in nums:
            target_a = 1.5 * b
            # Binary search for target_a in nums
            # bisect_left returns the index where target_a would be inserted to maintain order
            idx = bisect.bisect_left(nums, target_a)
            
            # Candidates are idx and idx-1
            # We check both to find the one with smaller |a - 1.5b|
            # Actually, we just need to check the closest ones.
            # The function |a/b - 1.5| is monotonic with respect to |a - 1.5b| for fixed b.
            # So the a closest to 1.5b gives the minimum error.
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                
                # Calculate error components
                # |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue # Invalid: exact 3/2 ratio
                
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                # num * best_den < best_num * den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b
                    
    # Final verification
    # Types must be Python int
    a = int(best_a)
    b = int(best_b)
    
    # Check constraints
    if a <= 0 or b <= 0 or a > N or b > N:
        return [11, 6]
        
    # Verify sigma equality (just in case)
    if sigma[a] != sigma[b]:
        return [11, 6]
        
    # Verify not exact 3/2
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [a, b]

# EVOLVE-BLOCK-END
