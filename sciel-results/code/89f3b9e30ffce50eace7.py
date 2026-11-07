"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    import time
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Determine the range we need to cover
    min_val = min(lo for lo, hi in gaps)
    max_val = max(hi for lo, hi in gaps)
    
    # We need LPF for integers in [min_val+1, max_val-1]
    # Actually we need to check m where lo < m < hi for each gap
    # Let's compute LPF for the entire range [min_val+1, max_val]
    
    start = min_val + 1
    end = max_val  # inclusive
    
    size = end - start + 1
    
    # Segmented sieve to compute least prime factor
    # First, get primes up to sqrt(max_val)
    from math import isqrt
    limit = isqrt(max_val) + 1
    
    # Simple sieve for small primes
    small_primes = []
    is_prime_small = [True] * (limit + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, limit + 1):
        if is_prime_small[i]:
            small_primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_prime_small[j] = False
    
    # Initialize LPF array
    # lpf[x] = least prime factor of x (for x in [start, end])
    lpf = list(range(start, end + 1))
    # Actually, let's use an array indexed by offset
    # lpf_arr[i] corresponds to value start + i
    
    lpf_arr = [0] * size
    for i in range(size):
        val = start + i
        lpf_arr[i] = val  # initially assume prime (LPF = itself)
    
    # For each small prime, mark its multiples in the range
    for p in small_primes:
        # Find the first multiple of p in [start, end]
        first = ((start + p - 1) // p) * p
        for m in range(first, end + 1, p):
            idx = m - start
            if lpf_arr[idx] > p:  # p is smaller than current LPF
                lpf_arr[idx] = p
    
    # Now for each gap, find witnesses
    for i, (lo, hi) in enumerate(gaps):
        gap_size = hi - lo
        # Look for m in (lo, hi) such that lpf[m] >= gap_size
        # m ranges from lo+1 to hi-1
        found = False
        for m in range(lo + 1, hi):
            idx = m - start
            if lpf_arr[idx] >= gap_size:
                result[i] = m
                found = True
                break
        if not found:
            result[i] = None
    
    return result
# EVOLVE-BLOCK-END
