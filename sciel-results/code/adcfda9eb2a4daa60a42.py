import time
import numpy as np

def construct():
    deadline = time.time() + 85.0  # 85s safety margin
    
    N = 1000000
    # Sieve for smallest prime factor up to N+1 (to handle n+1)
    limit = N + 1
    spf = np.arange(limit + 1, dtype=np.int32)
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers using SPF
    lpf_arr = np.ones(limit + 1, dtype=np.int32)
    for x in range(2, limit + 1):
        max_pf = 1
        val = x
        while val > 1:
            p = spf[val]
            max_pf = p
            while val % p == 0:
                val //= p
        lpf_arr[x] = max_pf
    
    best_n = 10000
    best_val = max(lpf_arr[10000], lpf_arr[10001])
    
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        ln = lpf_arr[n]
        if ln > best_val:
            continue  # can't improve
        ln1 = lpf_arr[n + 1]
        val = max(ln, ln1)
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
