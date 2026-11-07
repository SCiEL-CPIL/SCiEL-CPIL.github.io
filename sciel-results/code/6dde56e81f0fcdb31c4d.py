import time
import math

def construct():
    deadline = time.time() + 80  # 80s guard, 90s limit
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback: return a known valid pair
        return [11, 6]
    
    # Group numbers by sigma value
    # Only keep numbers with sigma value that appears more than once
    # First pass: count sigma values
    from collections import defaultdict
    groups = defaultdict(list)
    
    for n in range(1, N + 1):
        s = sigma[n]
        groups[s].append(n)
    
    # Check time
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Find best pair: minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group, sort and for each b, find a closest to 3b/2
    best_err = float('inf')
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 3:
            break
        
        nums.sort()
        k = len(nums)
        
        # For each b in nums, find a in nums closest to 3b/2
        # Use binary search
        for i in range(k):
            b = nums[i]
            target = 3.0 * b / 2.0  # ideal a value
            
            # Binary search for target in nums
            lo = 0
            hi = k
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1
            candidates = []
            if lo < k:
                candidates.append(nums[lo])
            if lo > 0:
                candidates.append(nums[lo - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # a/b = 3/2 exactly, invalid
                    continue
                den = 2 * b
                err = num / den
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
    
    # Final check: ensure a != b and sigma(a) == sigma(b) and a/b != 3/2
    if best_a == best_b or 2 * best_a == 3 * best_b:
        # Fallback
        return [11, 6]
    
    return [best_a, best_b]
