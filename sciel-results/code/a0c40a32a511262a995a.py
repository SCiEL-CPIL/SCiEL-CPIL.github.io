import numpy as np
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        # Fallback if sieve takes too long (unlikely, but safe)
        return [1, 1]

    # 2. Bucket indices by sigma value
    # This is O(N) but involves Python loop overhead.
    # Alternative: Use np.argsort on sigma and then group, but that's O(N log N).
    # Given N=10^7, the Python loop for bucketing might be slow.
    # Let's estimate: 10^7 iterations in Python is ~1s. This is acceptable.
    
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        buckets[int(sigma[i])].append(i)
        
    if time.time() > deadline:
        # Fallback: find any valid pair quickly
        for lst in buckets.values():
            if len(lst) >= 2:
                return [lst[0], lst[1]]
        return [1, 1]

    # 3. Search for best ratio within each bucket
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # Sort the buckets? No, just process each one.
    # We can prioritize buckets with larger sizes as they are more likely to contain close ratios.
    # But sorting buckets by size is O(B log B) where B is number of unique sigma values.
    # B is roughly N / log N or similar, so ~10^6. This is fine.
    
    # Actually, let's just iterate. The total work is dominated by the search inside buckets.
    
    for lst in buckets.values():
        if time.time() > deadline:
            break
            
        k = len(lst)
        if k < 2:
            continue
            
        # Sort the list for binary search
        lst.sort()
        
        # For each b in lst, find a in lst closest to 1.5 * b
        # We iterate through b.
        for b in lst:
            if time.time() > deadline:
                break
                
            target = 1.5 * b
            # Find insertion point
            pos = bisect.bisect_left(lst, target)
            
            # Check candidates at pos and pos-1
            # We need a != b
            candidates = []
            if pos < k:
                candidates.append(lst[pos])
            if pos > 0:
                candidates.append(lst[pos-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Error = |a/b - 1.5| = |2a - 3b| / (2b)
                # To avoid float division until comparison, we can compare:
                # |2a - 3b| / (2b) < best_err
                # |2a - 3b| < best_err * 2b
                # But best_err is float. Let's just compute float error.
                err = abs(2 * a - 3 * b) / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    if best_err == float('inf'):
        # Fallback: find any valid pair
        for lst in buckets.values():
            if len(lst) >= 2:
                best_a = lst[0]
                best_b = lst[1]
                break
    
    # Ensure types are Python int
    return [int(best_a), int(best_b)]
