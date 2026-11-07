"""Rewritten program: segmented-sieve LPF over the full range, then scan each gap."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _primes_upto(n):
    """Return list of primes <= n (Sieve of Eratosthenes)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            start = i * i
            step = i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    import time
    deadline = time.time() + 80.0  # safety margin vs 90s wall-clock

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # Determine the full range we need to cover
    lo_min = min(lo for lo, hi in gaps)
    hi_max = max(hi for lo, hi in gaps)
    # We need LPF for integers in [lo_min+1, hi_max-1]
    seg_lo = lo_min + 1
    seg_hi = hi_max - 1
    seg_len = seg_hi - seg_lo + 1

    if seg_len <= 0:
        return result

    # We need primes up to sqrt(hi_max) for the segmented sieve
    import math
    limit = int(math.isqrt(hi_max)) + 1
    small_primes = _primes_upto(limit)

    # Initialize LPF array: 0 means "not yet assigned" (will become the number itself if no factor found)
    # We'll store the least prime factor for each position in [seg_lo, seg_hi]
    # Use a list of ints; 0 = no factor found yet
    lpf = [0] * seg_len

    # For each small prime p, mark its multiples in the segment
    for p in small_primes:
        # Find first multiple of p in [seg_lo, seg_hi]
        first = ((seg_lo + p - 1) // p) * p
        if first > seg_hi:
            continue
        # Mark all multiples
        for m in range(first, seg_hi + 1, p):
            idx = m - seg_lo
            if lpf[idx] == 0:
                lpf[idx] = p

    # Now for each position where lpf is still 0, the number itself is prime (or 1)
    # Actually, if no prime <= sqrt(hi_max) divides m, then m is prime (since m < hi_max)
    # So lpf[m] = m in that case
    for i in range(seg_len):
        if lpf[i] == 0:
            lpf[i] = seg_lo + i  # the number itself is prime

    # Now scan each gap for a witness
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        # Need m in (lo, hi) i.e. m in [lo+1, hi-1] with lpf[m] >= gap
        # lo+1 corresponds to index 0 in our segment (since seg_lo = lo_min+1)
        # But we need to map correctly
        start_idx = lo + 1 - seg_lo
        end_idx = hi - 1 - seg_lo  # inclusive

        if start_idx < 0:
            start_idx = 0
        if end_idx >= seg_len:
            end_idx = seg_len - 1

        found = None
        for idx in range(start_idx, end_idx + 1):
            m = seg_lo + idx
            if lpf[idx] >= gap:
                found = m
                break

        if found is not None:
            result[i] = int(found)

        if time.time() > deadline:
            break

    # Ensure all entries are Python int or None
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
