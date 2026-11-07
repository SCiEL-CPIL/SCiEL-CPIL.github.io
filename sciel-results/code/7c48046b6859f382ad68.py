import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using SPF-based multiplicative sieve
    # sigma is multiplicative: if n = p1^a1 * p2^a2 * ... then sigma(n) = prod(sigma(pi^ai))
    # sigma(p^k) = (p^(k+1)-1)/(p-1)
    
    # Step 1a: Compute smallest prime factor (SPF) for all n
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            # i is prime
            spf[i::i] = i
    
    # Step 1b: Compute sigma using multiplicative property
    # sigma[n] = sigma[n // p^k] * (p^(k+1) - 1) / (p - 1) where p = spf[n]
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for n in range(2, N + 1):
        if time.time() > deadline - 5.0:
            break
        p = spf[n]
        m = n // p
        # Count power of p in n
        pk = p
        while m % p == 0:
            pk *= p
            m //= p
        # sigma(n) = sigma(m) * (pk*p - 1) // (p - 1)
        # pk*p = p^(k+1) where pk = p^k
        sigma[n] = sigma[m] * (pk * p - 1) // (p - 1)
    
    if time.time() > deadline - 2.0:
        # Fallback: return a known valid pair
        return [11, 6]
    
    # 2. Bucket indices by sigma value
    # Only include n where sigma[n] > 0 (all n >= 1)
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        s = int(sigma[i])
        if s > 0:
            buckets[s].append(i)
    
    if time.time() > deadline - 1.0:
        # Fallback: find any pair
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [int(lst[0]), int(lst[1])]
        return [11, 6]
    
    # 3. Search for best ratio
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Use exact rational comparison: num/den < best_num/best_den iff num*best_den < best_num*den
    
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6)  # = 4
    best_den = 2 * 6  # = 12
    
    # Sort all bucket lists once
    for key in buckets:
        if len(buckets[key]) >= 2:
            buckets[key].sort()
    
    if time.time() > deadline - 0.5:
        return [int(best_a), int(best_b)]
    
    # Process buckets
    for sig_val, lst in buckets.items():
        if time.time() > deadline - 0.5:
            break
        if len(lst) < 2:
            continue
        
        # For each b in lst, find a in lst closest to 1.5 * b
        # We can limit search: only consider a where |2a - 3b| / (2b) < best_num/best_den
        # i.e., |2a - 3b| < best_num * b / best_den
        # This gives a range for a: 
        # 3b - best_num*b/best_den < 2a < 3b + best_num*b/best_den
        # b*(3 - best_num/best_den)/2 < a < b*(3 + best_num/best_den)/2
        
        for b in lst:
            if time.time() > deadline - 0.5:
                break
                
            # Compute the range of a values that could beat current best
            # |2a - 3b| < best_num * b / best_den
            # Let threshold = best_num * b // best_den (integer division, conservative)
            threshold = (best_num * b) // best_den + 1  # +1 to be safe with integer div
            
            # a must satisfy: 3b - threshold < 2a < 3b + threshold
            # So: (3b - threshold + 1) // 2 <= a <= (3b + threshold - 1) // 2
            lo = max(1, (3 * b - threshold + 1) // 2)
            hi = min(N, (3 * b + threshold - 1) // 2)
            
            if lo > hi:
                continue
            
            # Binary search for candidates in [lo, hi]
            left_idx = bisect.bisect_left(lst, lo)
            right_idx = bisect.bisect_right(lst, hi)
            
            # Check candidates in [left_idx, right_idx)
            # But we only need to check a few around 1.5*b for efficiency
            # Actually, let's check all in the range since the range is narrow
            for idx in range(left_idx, right_idx):
                a = lst[idx]
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                den = 2 * b
                # Compare num/den with best_num/best_den using cross multiplication
                if num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
    
    # Ensure result is valid (not exact 3/2)
    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 4, 12
    
    return [int(best_a), int(best_b)]
