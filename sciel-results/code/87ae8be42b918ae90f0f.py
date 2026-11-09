"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def _build_lpf_table(limit, primes):
    """Build least prime factor table for all numbers up to limit using given primes."""
    lpf = [0] * (limit + 1)
    for p in primes:
        if p > limit:
            break
        for m in range(p, limit + 1, p):
            if lpf[m] == 0:
                lpf[m] = p
    return lpf


def construct():
    start_time = time.time()
    deadline = start_time + 80  # leave 10s safety margin

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # Find max gap size to determine sieve limit
    max_gap = max(hi - lo for lo, hi in gaps)

    # We need primes < max_gap to check divisibility
    # LPF(m) >= gap_size means m is not divisible by any prime < gap_size
    sieve_limit = max_gap  # primes up to max_gap
    small_primes_all = _sieve_primes_up_to(sieve_limit)

    # Precompute LPF table for the range of all candidates
    min_lo = min(lo for lo, hi in gaps)
    max_hi = max(hi for lo, hi in gaps)
    
    # Build LPF table for the entire range
    # We need to check LPF for numbers in [min_lo+1, max_hi-1]
    # But we only need to know if LPF >= gap_size
    # A number m has LPF >= k iff m is not divisible by any prime < k
    # So we can check: for each m, find the smallest prime p < gap_size that divides m
    
    # Alternative approach: for each gap, check candidates against primes < gap_size
    # This is what the original does. Let's optimize by precomputing divisibility.
    
    # For each prime p in small_primes_all, mark which candidates in the full range are divisible by p
    # Then for each gap, a candidate m is valid iff it's not marked by any prime < gap_size
    
    # Build a 2D structure: for each prime index, a set of positions divisible by that prime
    # But that might be too memory-heavy. Let's try a different approach.
    
    # For each candidate position (relative to min_lo), track the smallest prime that divides it
    # This is essentially the LPF table but offset.
    
    # Since the range is small (~280), we can just do direct checks efficiently.
    # Precompute for each candidate m (absolute value) the smallest prime < max_gap that divides m.
    # Then for gap with size g, m is valid iff lpf_small[m] >= g or lpf_small[m] == 0 (no small prime divides it)
    
    # Build a dictionary: for each m in [min_lo+1, max_hi-1], find smallest prime in small_primes_all that divides m
    # Actually, we need: smallest prime p < max_gap such that p | m. If none exists, LPF_small = infinity.
    
    range_start = min_lo + 1
    range_end = max_hi - 1
    range_len = range_end - range_start + 1
    
    # lpf_small[m - range_start] = smallest prime < max_gap that divides m, or 0 if none
    lpf_small = [0] * range_len
    
    for p in small_primes_all:
        if p >= max_gap:
            break
        # Find all multiples of p in [range_start, range_end]
        first_mult = ((range_start + p - 1) // p) * p
        for m in range(first_mult, range_end + 1, p):
            idx = m - range_start
            if lpf_small[idx] == 0:
                lpf_small[idx] = p
    
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break

        gap_size = hi - lo

        found = None
        for m in range(lo + 1, hi):
            idx = m - range_start
            # m is valid if no prime < gap_size divides it
            # i.e., lpf_small[idx] == 0 or lpf_small[idx] >= gap_size
            if lpf_small[idx] == 0 or lpf_small[idx] >= gap_size:
                found = m
                break
        result[i] = found

    # Ensure all entries are Python int or None (not numpy types)
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
