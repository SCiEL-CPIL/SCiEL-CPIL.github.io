import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s
    
    # Compute sigma(n) for n up to N
    N = 5_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 500_000 == 0:
            if time.time() > deadline - 10:
                break
    
    # Build groups: sigma value -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        # Fallback: return a known valid pair
        return [11, 6]
    
    # Find best pair minimizing |a/b - 3/2| exactly
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # Compare: |2a1 - 3b1|/b1 vs |2a2 - 3b2|/b2
    # i.e., |2a1 - 3b1| * b2 vs |2a2 - 3b2| * b1
    
    best_num = None  # |2a - 3b|
    best_den = None  # b
    best_a = None
    best_b = None
    
    for s_val, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 3:
            break
        
        nums.sort()
        # For each b in nums, find a in nums such that a/b ≈ 3/2
        # i.e., a ≈ 1.5 * b
        for b in nums:
            ideal_a = (3 * b) // 2  # floor of 1.5*b
            # Search around ideal_a in nums
            # Use binary search
            lo = 0
            hi = len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < ideal_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check positions lo and lo-1
            for idx in [lo - 1, lo]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    # Compute |2a - 3b|
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue  # ratio exactly 3/2, invalid
                    # Compare with best: diff/b vs best_num/best_den
                    # diff * best_den vs best_num * b
                    if best_num is None:
                        best_num = diff
                        best_den = b
                        best_a = a
                        best_b = b
                    else:
                        left = diff * best_den
                        right = best_num * b
                        if left < right:
                            best_num = diff
                            best_den = b
                            best_a = a
                            best_b = b
    
    if best_a is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]
