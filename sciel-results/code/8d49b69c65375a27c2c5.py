"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    # Determine the range of values we need to check
    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])
    
    # We need to check integers in [all_lo+1, all_hi-1]
    # For each such integer m, we need its least prime factor (LPF)
    # LPF[m] >= gap_size means m is either prime or its smallest factor is >= gap_size
    
    # Precompute primes up to sqrt(all_hi) for trial division, and also up to max_gap
    # Actually, we need LPF for each m. Let's use a segmented sieve approach.
    
    # The range is small: 10008 to 10288, about 280 numbers
    # For each m in that range, compute its least prime factor by trial division
    # using precomputed small primes
    
    max_val = all_hi
    min_val = all_lo
    
    # Precompute primes up to sqrt(max_val) for efficient trial division
    from math import isqrt
    limit = isqrt(max_val) + 1
    # Simple sieve for primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Also, for gaps where gap_size > limit, we need primes up to gap_size
    # The max gap is 10289-10273 = 16, so gap sizes are small (2 to 16)
    # Actually let me check: max gap = max(hi-lo for lo,hi in P['gaps'])
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need primes up to max(max_gap, isqrt(max_val)) for LPF computation
    # But actually, for LPF, we just need to try dividing by primes in order
    # If no prime <= sqrt(m) divides m, then LPF[m] = m (m is prime)
    # If a prime p <= sqrt(m) divides m, then LPF[m] = p (the first such)
    # But we also need to handle the case where m has a prime factor between sqrt(m) and m
    # That can't happen - if m is composite, it has a factor <= sqrt(m)
    # So LPF[m] is either a prime <= sqrt(m) or m itself (if m is prime)
    
    # So for each m, we trial divide by small_primes (which go up to isqrt(max_val))
    # If none divide m, LPF[m] = m
    
    # Compute LPF for all m in [min_val+1, max_val-1]
    lpf = {}
    for m in range(min_val + 1, max_val):
        found = False
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                lpf[m] = p
                found = True
                break
        if not found:
            lpf[m] = m  # m is prime
    
    # Now for each gap, find a witness m with lo < m < hi and lpf[m] >= (hi - lo)
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Scan from both ends to find a valid witness quickly
        # Try middle first, then expand outward
        mid = (lo + hi) // 2
        # Actually, just scan all candidates in the gap
        for m in range(lo + 1, hi):
            if lpf.get(m, m) >= gap:
                result[i] = m
                break
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
