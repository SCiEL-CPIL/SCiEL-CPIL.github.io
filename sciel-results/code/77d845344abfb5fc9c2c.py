import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using SPF-based multiplicative sieve
    # Step 1a: Compute smallest prime factor (SPF) for all n
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Step 1b: Compute sigma using multiplicative property
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for n in range(2, N + 1):
        if time.time() > deadline - 10.0:
            break
        p = spf[n]
        m = n // p
        pk = p
        while m % p == 0:
            pk *= p
            m //= p
        sigma[n] = sigma[m] * (pk * p - 1) // (p - 1)
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # 2. Bucket indices by sigma value
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        s = int(sigma[i])
        if s > 0:
            buckets[s].append(i)
    
    if time.time() > deadline - 3.0:
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [int(lst[0]), int(lst[1])]
        return [11, 6]
    
    # 3. Search for best ratio
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Use exact rational comparison
    
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6)  # = 4
    best_den = 2 * 6  # = 12
    
    # Sort all bucket lists
    for key in buckets:
        if len(buckets[key]) >= 2:
            buckets[key].sort()
    
    if time.time() > deadline - 1.0:
        return [int(best_a), int(best_b)]
    
    # Process buckets
    for sig_val, lst in buckets.items():
        if time.time() > deadline - 0.5:
            break
        if len(lst) < 2:
            continue
        
        for b in lst:
            if time.time() > deadline - 0.5:
                break
            
            threshold = (best_num * b) // best_den + 1
            
            lo = max(1, (3 * b - threshold + 1) // 2)
            hi = min(N, (3 * b + threshold - 1) // 2)
            
            if lo > hi:
                continue
            
            left_idx = bisect.bisect_left(lst, lo)
            right_idx = bisect.bisect_right(lst, hi)
            
            for idx in range(left_idx, right_idx):
                a = lst[idx]
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
    
    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 4, 12
    
    return [int(best_a), int(best_b)]
