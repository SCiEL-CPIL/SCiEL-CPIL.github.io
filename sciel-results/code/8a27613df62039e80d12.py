"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    
    # Determine the range we need to cover
    # We need LPF for all integers in [min_lo+1, max_hi-1]
    # min_lo = 10007, max_hi = 10289
    # So we need LPF for [10008, 10288]
    
    min_val = min(lo for lo, hi in P['gaps']) + 1  # 10008
    max_val = max(hi for lo, hi in P['gaps']) - 1  # 10288
    
    # Segmented Sieve for Least Prime Factor in [min_val, max_val]
    # Step 1: Find all primes up to sqrt(max_val)
    limit = isqrt(max_val)
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Initialize LPF array for the segment [min_val, max_val]
    # lpf[i] corresponds to number (min_val + i)
    # 0 means "no factor found yet" (will be prime if 0 at end)
    size = max_val - min_val + 1
    lpf = [0] * size
    
    # Step 3: Mark least prime factors
    for p in primes:
        # Find the first multiple of p in [min_val, max_val]
        # start = ceil(min_val / p) * p
        start = ((min_val + p - 1) // p) * p
        # If start is p itself, we should start from p*p for standard sieve logic?
        # No, for LPF in a segment, we just need to mark multiples.
        # However, if start == p, then p is in the segment. But p <= sqrt(max_val) < min_val usually.
        # Here min_val ~ 10000, p <= 101. So start will be > p.
        for multiple in range(start, max_val + 1, p):
            idx = multiple - min_val
            if lpf[idx] == 0:
                lpf[idx] = p
    
    # Now for each gap, find a witness m with lo < m < hi and LPF(m) >= gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            idx = m - min_val
            # If lpf[idx] is 0, m is prime, so LPF(m) = m
            # Otherwise, LPF(m) = lpf[idx]
            lpf_m = m if lpf[idx] == 0 else lpf[idx]
            
            if lpf_m >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
