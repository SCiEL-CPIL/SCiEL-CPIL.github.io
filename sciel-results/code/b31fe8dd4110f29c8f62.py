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
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    deadline = time.monotonic() + 80.0  # safety margin before 90s kill
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to sqrt of max possible value for LPF verification
    max_val = max(hi for _, hi in P['gaps'])
    primes = _sieve_primes_up_to(isqrt(max_val))
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.monotonic() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue
        
        # Find candidates strictly between lo and hi whose LPF >= gap
        # A candidate m has LPF < gap iff m is divisible by some prime < gap
        # So we need m not divisible by any prime < gap
        
        # Mark all numbers in (lo, hi) that are divisible by some prime < gap
        # Use a bytearray for the range
        size = hi - lo - 1
        if size <= 0:
            continue
        
        marked = bytearray(size)  # marked[j] = 1 means lo+1+j is divisible by some prime < gap
        
        # For each prime p < gap, mark multiples in (lo, hi)
        for p in primes:
            if p >= gap:
                break
            # Find first multiple of p in (lo, hi)
            # We need lo < m < hi, so m in [lo+1, hi-1]
            # First multiple >= lo+1:
            start = ((lo + 1 + p - 1) // p) * p
            if start < lo + 1:
                start += p
            # Mark multiples from start to hi-1
            for m in range(start, hi, p):
                marked[m - lo - 1] = 1
        
        # Now find unmarked candidates - these have LPF >= gap
        # Search from both ends toward the middle for efficiency
        found = False
        # Try from left
        for j in range(size):
            if not marked[j]:
                m = lo + 1 + j
                # Verify LPF >= gap (safety check)
                lpf = m
                for p in primes:
                    if p * p > m:
                        break
                    if m % p == 0:
                        lpf = p
                        break
                if lpf >= gap:
                    result[i] = int(m)
                    found = True
                    break
            if time.monotonic() > deadline:
                break
        
        if not found:
            # Try from right
            for j in range(size - 1, -1, -1):
                if not marked[j]:
                    m = lo + 1 + j
                    lpf = m
                    for p in primes:
                        if p * p > m:
                            break
                        if m % p == 0:
                            lpf = p
                            break
                    if lpf >= gap:
                        result[i] = int(m)
                        found = True
                        break
                if time.monotonic() > deadline:
                    break
    
    # Ensure types are correct
    return result
# EVOLVE-BLOCK-END
