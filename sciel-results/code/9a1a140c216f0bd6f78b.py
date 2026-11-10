"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _powerful_part(n, spf):
    """Return the powerful part of n: product of p^e for each prime p where exponent e >= 2."""
    if n <= 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e >= 2:
            result *= p ** e
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 1000000
    # Build smallest prime factor sieve up to N+1
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all numbers in [10000, N+1]
    # We need powerful_part(n) for n in [10000, N] and powerful_part(n+1) for n in [10000, N]
    # So we need pp[10000] through pp[N+1]
    pp = [0] * (N + 2)
    for i in range(10000, N + 2):
        pp[i] = _powerful_part(i, spf)
    
    # Find the n in [10000, N] that maximizes pp[n] * pp[n+1]
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, N):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
