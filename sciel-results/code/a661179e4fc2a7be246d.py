"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(limit):
    """Return a list where sieve[p] is True iff p is prime, for p in [0, limit]."""
    if limit < 2:
        return [False] * (limit + 1)
    sieve = [True] * (limit + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return sieve

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Precompute primes up to a reasonable bound using sieve
    # The largest number we might test is ~10289, so sieve up to 10300
    max_val = max(hi for _, hi in gaps) + 100
    sieve = _sieve(max_val)
    
    # Precompute least prime factor (lpf) for all numbers up to max_val
    # lpf[x] = smallest prime factor of x, or x if x is prime
    lpf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    # For each gap, find a witness m strictly between lo and hi
    # such that lpf[m] >= (hi - lo)
    # Strategy: for each gap, try numbers in the gap. To maximize success,
    # we want to find at least one valid m per gap.
    
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        # Try all numbers in (lo, hi)
        # We want lpf[m] >= gap_size
        # A number m has lpf[m] >= gap_size iff m has no prime factor < gap_size
        # i.e., m is either prime, or all its prime factors are >= gap_size
        # For small gaps, this is easier.
        
        found = False
        # Try from both ends inward? Or just scan?
        # Let's scan all candidates in the gap.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                result[i] = m
                found = True
                break
        
        if not found:
            # Try a more targeted search: look for numbers with large least prime factor
            # This is harder; for now, leave as None
            pass
    
    # Ensure all values are Python int (not numpy)
    # They should already be Python ints from range()
    
    return result
# EVOLVE-BLOCK-END
