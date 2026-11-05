import time
import math

def construct():
    N = 1000000
    # We need P(n) for n up to N, and P(n+1) for n up to N-1, so up to N.
    # Actually n is in [10000, 1000000]. So we need P up to 1000000.
    # If n=1000000, we need P(1000000) and P(1000001).
    # So size should be 1000001 + 1 = 1000002.
    
    limit = 1000001
    
    # Sieve of Eratosthenes for Smallest Prime Factor
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts
    P = [1] * (limit + 1)
    for i in range(2, limit + 1):
        m = i
        res = 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        P[i] = res
        
    # Find max product
    best_n = 10000
    best_val = -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    for n in range(10000, 1000000):
        if time.time() > deadline:
            break
        val = P[n] * P[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
