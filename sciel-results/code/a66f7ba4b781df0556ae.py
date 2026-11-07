"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    
    # Determine the range we need to cover
    min_val = min(lo for lo, hi in P['gaps']) + 1  # 10008
    max_val = max(hi for lo, hi in P['gaps']) - 1  # 10288
    
    # Segmented sieve: compute LPF for all integers in [min_val, max_val]
    limit = isqrt(max_val)
    
    # Step 1: Find all primes up to limit using simple sieve
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Segmented sieve - for each number in [min_val, max_val],
    # find its least prime factor
    size = max_val - min_val + 1
    # lpf[i - min_val] stores the least prime factor of (i), or 0 if not yet found
    lpf = [0] * size
    
    for p in primes:
        # Find the first multiple of p in [min_val, max_val]
        start = ((min_val + p - 1) // p) * p
        if start < p * p:
            # If start < p^2, we might miss that p itself could be the LPF
            # for p if p is in range, but p <= limit < min_val typically
            # Actually p <= 101 and min_val = 10008, so start >= min_val
            pass
        for j in range(start, max_val + 1, p):
            idx = j - min_val
            if lpf[idx] == 0:  # Only set if not already marked (keep smallest)
                lpf[idx] = p
    
    # Step 3: For each gap, find a witness m with lo < m < hi and lpf[m] >= gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            idx = m - min_val
            # If lpf[idx] is 0, m is prime, so LPF = m
            # Otherwise, LPF = lpf[idx]
            lpf_val = m if lpf[idx] == 0 else lpf[idx]
            if lpf_val >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
