import numpy as np
from collections import defaultdict
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin

    # 1. Compute sigma(n)
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Divisor sieve
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline:
            break

    # If we broke early, sigma is incomplete. 
    # But the sieve is roughly uniform in work. 
    # If we break, we might miss some. 
    # However, the early part (small i) contributes the most to the sum for many j.
    # Actually, small i adds to many j. Large i adds to few j.
    # If we break at i=K, we have added divisors 1..K.
    # We are missing divisors > K.
    # This makes sigma(n) too small.
    # This would lead to incorrect grouping.
    # So we must ensure the sieve completes or use a faster method.
    
    # Let's assume it completes. If not, we return seed.
    if time.time() > deadline:
        return [11, 6]

    # 2. Group indices by sigma value
    groups = defaultdict(list)
    for i in range(1, N + 1):
        if time.time() > deadline:
            return [11, 6]
        s = sigma[i]
        groups[s].append(i)

    # 3. Find best pair
    best_err = 1.0 / 3.0  # Seed error
    best_a, best_b = 11, 6

    for s, nums in groups.items():
        if time.time() > deadline:
            break
        if len(nums) < 2:
            continue
        
        # Sort nums to allow pruning/binary search
        nums.sort()
        K = len(nums)
        
        # If K is small, check all pairs
        # If K is large, use binary search
        if K < 50:
            for i in range(K):
                a = nums[i]
                for j in range(K):
                    b = nums[j]
                    if a == b: continue
                    # Pruning: we want a/b close to 1.5
                    # |a/b - 1.5| < best_err
                    # a/b in [1.5 - best_err, 1.5 + best_err]
                    # a in b * [1.5 - best_err, 1.5 + best_err]
                    # Since best_err starts at 0.333, range is [1, 2].
                    # So a must be between b and 2b.
                    if a < b:
                        if a > 2 * b: continue
                    else:
                        if b > 2 * a: continue
                        
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
        else:
            # Binary search approach
            for a in nums:
                # We want b such that a/b approx 1.5 => b approx 2a/3
                target_b = 2 * a / 3
                # Find b in nums closest to target_b
                # Also check b such that b/a approx 1.5 => a approx 1.5 b => b approx 2/3 a
                # The condition |a/b - 1.5| is symmetric in a way? No.
                # |a/b - 1.5| = |b/a * (a/b)^2 ...| No.
                # Just find b closest to 2a/3.
                
                # Use bisect
                import bisect
                pos = bisect.bisect_left(nums, target_b)
                
                candidates = []
                if pos < K: candidates.append(nums[pos])
                if pos > 0: candidates.append(nums[pos-1])
                
                # Also, we might want a/b to be 1.5, so b = 2a/3.
                # But what if a/b is slightly different?
                # The function f(b) = |a/b - 1.5| is monotonic for b < 2a/3 and b > 2a/3.
                # So the closest b to 2a/3 is the best for this a.
                
                for b in candidates:
                    if a == b: continue
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        
    return [int(best_a), int(best_b)]
