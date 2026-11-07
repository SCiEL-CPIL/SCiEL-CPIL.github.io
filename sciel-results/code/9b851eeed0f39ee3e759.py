"""Rewritten program: SIEVE-BASED witness search over the 32 prime gaps."""
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
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    import time
    deadline = time.time() + 85.0  # safety margin vs 90s wall-clock

    gaps = P['gaps']
    result = [None] * len(gaps)

    # Precompute all primes up to max gap - 1
    max_gap = max(hi - lo for lo, hi in gaps)
    all_primes = _primes_upto(max_gap - 1)

    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        # Need m in (lo, hi) with least prime factor >= gap,
        # i.e. m not divisible by any prime < gap.
        small = [p for p in all_primes if p < gap]
        
        # Sieve-based approach: mark candidates that are NOT divisible by any small prime
        # Use a bytearray to track which candidates are "clean" (not divisible by any p < gap)
        # Candidate range: lo+1 to hi-1
        num_candidates = hi - lo - 1
        if num_candidates <= 0:
            continue
        
        # Initialize all candidates as potentially valid
        clean = bytearray([1]) * num_candidates
        
        for p in small:
            # Mark all multiples of p in the candidate range as not clean
            # First multiple of p >= lo+1
            first = ((lo + 1 + p - 1) // p) * p
            if first > hi - 1:
                continue
            # Mark indices corresponding to first, first+p, first+2p, ...
            start_idx = first - (lo + 1)
            for j in range(start_idx, num_candidates, p):
                clean[j] = 0
        
        # Find the first clean candidate
        found = None
        for j in range(num_candidates):
            if clean[j]:
                found = int(lo + 1 + j)
                break
        
        if found is not None:
            result[i] = found
        
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
