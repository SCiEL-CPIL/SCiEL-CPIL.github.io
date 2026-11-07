import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [1, 1] # Fallback, though this shouldn't happen

    # 2. Bucket indices by sigma value
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        buckets[sigma[i]].append(i)
        
    if time.time() > deadline:
        # Try to return something valid? We need sigma(a)=sigma(b).
        # If we haven't sorted, we can't search efficiently.
        # But bucketing is done. We can just pick any pair from a bucket with size > 1.
        # But we want min error. Without sorting, we can't find min error easily.
        # However, bucketing is O(N). If we are here, we spent time on sieve.
        # Let's assume we have time.
        pass

    # 3. Sort lists in buckets
    for key in buckets:
        buckets[key].sort()

    if time.time() > deadline:
        # Fallback: find any pair
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [lst[0], lst[1]]
        return [1, 1]

    # 4. Search for best ratio
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # We iterate b. We can optimize by noting that if b is very small or large, 
    # the potential a is constrained.
    # But simple iteration is fine.
    
    for b in range(1, N + 1):
        if time.time() > deadline:
            break
            
        s_b = sigma[b]
        lst = buckets.get(s_b)
        
        if not lst or len(lst) < 2:
            continue
            
        target = 1.5 * b
        idx = bisect.bisect_left(lst, target)
        
        # Check neighbors
        # We want a != b
        for cand_idx in (idx - 1, idx):
            if 0 <= cand_idx < len(lst):
                a = lst[cand_idx]
                if a == b:
                    continue
                # Calculate error exactly? 
                # |a/b - 1.5| = |2a - 3b| / (2b)
                # We can compare |2a - 3b| / b to avoid float issues?
                # Float is fine for comparison, but let's use integer math for precision if needed.
                # err = abs(2*a - 3*b) / (2.0 * b)
                err = abs(2 * a - 3 * b) / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]
