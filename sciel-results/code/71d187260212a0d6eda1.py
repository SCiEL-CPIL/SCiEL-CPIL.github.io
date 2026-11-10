"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    MIN_N = 10000
    MAX_N = 1000000
    
    # Build smallest prime factor sieve
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all n in [1, MAX_N]
    # powerful_part(n) = product of p^e where e >= 2 in factorization of n
    pp = [1] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        if spf[n] == n:
            # n is prime, powerful part is 1
            pp[n] = 1
        else:
            p = spf[n]
            m = n
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                pp[n] = (p ** e) * pp[m]
            else:
                pp[n] = pp[m]
    
    # Find best n in [MIN_N, MAX_N-1] maximizing pp[n] * pp[n+1]
    best_n = MIN_N
    best_product = 0
    
    for n in range(MIN_N, MAX_N):
        prod = pp[n] * pp[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
