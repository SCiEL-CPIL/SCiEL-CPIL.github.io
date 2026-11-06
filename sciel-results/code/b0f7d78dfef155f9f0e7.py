"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part(n, spf):
    """Product of p^e for all p^e || n with e >= 2, using SPF table."""
    if n <= 1:
        return 1
    pp = 1
    temp = n
    while temp > 1:
        p = spf[temp]
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    spf = _build_spf(N)
    
    # Compute powerful part for all numbers
    pp = [1] * (N + 1)
    for i in range(2, N + 1):
        pp[i] = _powerful_part(i, spf)
    
    # Find max product pp[n] * pp[n+1] for n in [10000, 1000000]
    best_n = 10000
    best_prod = pp[10000] * pp[10001]
    
    for n in range(10000, N):
        prod = pp[n] * pp[n + 1]
        if prod > best_prod:
            best_prod = prod
            best_n = n
        
        # Time check
        if (n - 10000) % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n

# EVOLVE-BLOCK-END
