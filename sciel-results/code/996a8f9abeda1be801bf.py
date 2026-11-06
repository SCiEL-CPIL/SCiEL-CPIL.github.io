"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _build_spf(limit):
    """Build smallest prime factor array up to limit."""
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part(n, spf):
    """Compute powerful part of n: product of p^e for e >= 2 in factorization."""
    if n <= 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        if count >= 2:
            result *= p ** count
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    lo, hi = 10000, 1000000
    limit = hi + 1  # need to factor up to 1000001
    
    # Build SPF
    spf = _build_spf(limit)
    
    # Precompute powerful parts for all n in [lo, hi+1]
    pp = [0] * (limit + 1)
    for n in range(lo, limit + 1):
        pp[n] = _powerful_part(n, spf)
        if time.time() > deadline:
            break
    
    # Find max product
    best_n = lo
    best_val = pp[lo] * pp[lo + 1] if lo + 1 <= limit else 0
    
    for n in range(lo, hi):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
