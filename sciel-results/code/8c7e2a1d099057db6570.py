import time
import sys
import bisect
from fractions import Fraction

def construct():
    N = 10_000_000
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Phase 1: Compute sigma(n) for all n up to N
    # sigma(n) = sum of all positive divisors of n
    sigma = [0] * (N + 1)
    
    # Sieve-like computation: for each d, add d to sigma(m) for all m that are multiples of d
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
    
    if time.time() > deadline - 10:
        # Not enough time to finish; return a known valid pair
        return [11, 6]
    
    # Phase 2: Group numbers by their sigma value
    # We need pairs (a, b) with sigma(a) = sigma(b) and a != b
    # To minimize |a/b - 3/2|, for each group we want a/b close to 1.5
    
    # Build groups: sigma_value -> sorted list of n's
    from collections import defaultdict
    groups = defaultdict(list)
    
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline - 8:
        return [11, 6]
    
    # Phase 3: For each group, find the pair (a, b) that minimizes |a/b - 3/2|
    # Within a group, all have the same sigma. We want a/b ≈ 3/2, i.e., a ≈ 1.5*b
    # For each b in the group, the best a is the one closest to 1.5*b
    # Since the group is sorted, we can use binary search
    
    best_err = Fraction(1, 1)  # Use exact rational comparison
    best_pair = (11, 6)
    
    # Sort groups by size descending to check larger groups first (more likely to have good pairs)
    # Actually, let's just iterate all groups
    
    for s_val, nums in groups.items():
        if time.time() > deadline - 5:
            break
        if len(nums) < 2:
            continue
        
        nums.sort()
        n = len(nums)
        
        # For each b, find a closest to 1.5*b
        # Use two-pointer or binary search
        # Since we want a/b ≈ 1.5, a ≈ 1.5*b
        # For each b, bisect in nums to find 1.5*b
        
        # We'll iterate over b values and find the best a
        # To be efficient, iterate over smaller side if possible
        # Actually, both a and b range over the same set, so we can't easily split
        
        # For each b in nums, find a in nums closest to 1.5*b
        # Use bisect to find insertion point of 1.5*b
        
        for b in nums:
            if time.time() > deadline - 3:
                break
            
            target = 1.5 * b
            # Find the position where target would be inserted
            idx = bisect.bisect_left(nums, target)
            
            candidates = []
            if idx > 0:
                candidates.append(nums[idx - 1])
            if idx < n:
                candidates.append(nums[idx])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute exact error |a/b - 3/2| = |2a - 3b| / (2b)
                # This is exact rational arithmetic
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = Fraction(num, den)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    if best_err == 0:
                        return [a, b]
    
    # Return as Python int list
    return [int(best_pair[0]), int(best_pair[1])]
