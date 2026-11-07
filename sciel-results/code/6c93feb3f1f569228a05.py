import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Linear Sieve for Smallest Prime Factor (SPF)
    spf = np.zeros(N + 1, dtype=np.int64)
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            
    # 2. Compute sigma(n) using DP
    sigma = np.zeros(N + 1, dtype=np.int64)
    pow_p = np.zeros(N + 1, dtype=np.int64)
    sigma_part = np.zeros(N + 1, dtype=np.int64)
    sigma_rest = np.zeros(N + 1, dtype=np.int64)
    
    sigma[1] = 1
    pow_p[1] = 1
    sigma_part[1] = 1
    sigma_rest[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p != 0:
            # n = p * m, gcd(m, p) = 1
            pow_p[n] = p
            sigma_part[n] = 1 + p
            sigma_rest[n] = sigma[m]
        else:
            # n = p^k * u, m = p^(k-1) * u
            pow_p[n] = pow_p[m] * p
            sigma_part[n] = sigma_part[m] + pow_p[n]
            sigma_rest[n] = sigma_rest[m]
        sigma[n] = sigma_part[n] * sigma_rest[n]
        
        if time.time() > deadline - 10.0:
            # Fallback if too slow, though unlikely
            break

    # 3. Group n by sigma(n)
    d = defaultdict(list)
    for n in range(1, N + 1):
        d[sigma[n]].append(n)
        
    # 4. Search for best pair
    best_err = float('inf')
    best_a, best_b = 1, 1
    
    for L in d.values():
        if len(L) < 2:
            continue
            
        # L is sorted
        len_L = len(L)
        
        # Two pointer / binary search approach
        # For each b, find a ~ 1.5 b
        # Since L is sorted, we can use bisect
        import bisect
        
        for i in range(len_L):
            b = L[i]
            target = 1.5 * b
            
            # Find index of first element >= target
            # We want a > b, so we search in L[i+1:]
            # bisect_left returns index in whole list
            idx = bisect.bisect_left(L, target, i + 1)
            
            # Check candidate at idx
            if idx < len_L:
                a = L[idx]
                # Check if a/b is exactly 1.5
                if 2 * a != 3 * b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
            
            # Check candidate at idx - 1
            # Must be > b, so idx-1 > i
            if idx - 1 > i:
                a = L[idx - 1]
                if 2 * a != 3 * b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        
    return [int(best_a), int(best_b)]
