import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 10_000_000
    
    # Linear sieve to compute sigma
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    is_composite = [False] * (N + 1)
    spf_power = [0] * (N + 1)
    spf_sigma = [0] * (N + 1)
    
    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            spf_power[i] = i
            spf_sigma[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            if i % p == 0:
                spf_power[i * p] = spf_power[i] * p
                spf_sigma[i * p] = spf_sigma[i] + spf_power[i * p]
                sigma[i * p] = (sigma[i] // spf_sigma[i]) * spf_sigma[i * p]
                break
            else:
                spf_power[i * p] = p
                spf_sigma[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(2, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Find best pair
    best_diff = float('inf')
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline:
            break
        nums.sort()
        # For each b, find a closest to 3*b/2
        for b in nums:
            target = 3 * b // 2
            # Binary search for closest to target
            lo, hi = 0, len(nums) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            candidates = []
            if lo < len(nums):
                candidates.append(nums[lo])
            if lo > 0:
                candidates.append(nums[lo - 1])
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Invalid
                # We want to minimize diff / (2*b)
                # For comparison, minimize diff * 2 / b ... actually just track the rational
                # To avoid floating point, compare diff_a / b_a vs diff_b / b_b
                # i.e., diff_a * b_b vs diff_b * b_a
                # But for simplicity, use float for now (values are small enough)
                val = diff / (2.0 * b)
                if val < best_diff:
                    best_diff = val
                    best_a = a
                    best_b = b
    
    return [int(best_a), int(best_b)]
