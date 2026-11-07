"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    # Safety margin: The task says 60s, inspirations used 70-75s for a 90s limit.
    # We will use a generous 50s limit to be safe against any hidden overhead.
    deadline = start_time + 50

    # Precompute smallest prime factor for numbers up to max candidate + 1
    # The candidates are strictly between lo and hi, so max candidate is max(hi) - 1
    max_candidate = max(hi for lo, hi in P['gaps'])
    
    # SPF sieve
    spf = list(range(max_candidate + 1))
    if max_candidate >= 0:
        spf[0] = 0
    if max_candidate >= 1:
        spf[1] = 1
        
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def least_prime_factor(m):
        """Return the least prime factor of m using precomputed SPF table."""
        if m <= max_candidate:
            return spf[m]
        # Fallback for larger m (shouldn't happen given our range)
        for d in range(2, isqrt(m) + 1):
            if m % d == 0:
                return d
        return m

    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap
        # Linear scan is sufficient and robust for these small gaps
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            lpf = least_prime_factor(m)
            if lpf >= gap:
                result[i] = int(m)
                break
    
    return result
# EVOLVE-BLOCK-END
