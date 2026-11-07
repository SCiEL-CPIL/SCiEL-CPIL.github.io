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
    
    # Segmented sieve to compute LPF for all integers in [min_val, max_val]
    # First, compute primes up to sqrt(max_val)
    limit = isqrt(max_val)
    
    # Simple sieve for primes up to limit
    is_prime_small = [True] * (limit + 1)
    is_prime_small[0] = False
    is_prime_small[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime_small[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_small[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime_small[i]]
    
    # Initialize LPF array for the segment [min_val, max_val]
    # lp_seg[k] will store the least prime factor of (min_val + k), or (min_val + k) if prime
    seg_len = max_val - min_val + 1
    lp_seg = list(range(min_val, max_val + 1))  # lp_seg[k] = min_val + k initially
    
    # For each prime p, mark multiples in the segment
    for p in primes:
        # Find the first multiple of p in [min_val, max_val]
        start = ((min_val + p - 1) // p) * p
        if start < p * p:
            # If start < p*p, we might have already marked it with a smaller prime
            # But for LPF, we only want to mark if lp_seg is still the number itself
            # Actually, we need to be careful: we want the LEAST prime factor
            # So we should only set lp_seg if it's still the original value (i.e., not yet marked by a smaller prime)
            pass
        
        for m in range(start, max_val + 1, p):
            idx = m - min_val
            # Only set if not already set to a smaller prime factor
            if lp_seg[idx] == m:
                lp_seg[idx] = p
    
    # Now lp_seg[k] contains the least prime factor of (min_val + k)
    # For primes in the range, lp_seg[k] will still be (min_val + k) which is correct
    
    # For each gap, find a witness m with lo < m < hi and lp[m] >= gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            idx = m - min_val
            if lp_seg[idx] >= gap:
                result[i] = m
                break
    
    # Ensure all entries are Python int or None
    # The values in result are already Python ints from range()
    
    return result
# EVOLVE-BLOCK-END
