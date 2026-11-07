# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _compute_lpf(limit):
    """
    Compute least prime factor for all integers up to limit.
    lpf[1] = 1 (or 0, doesn't matter for our range)
    lpf[p] = p for prime p
    lpf[n] = smallest prime factor of n for composite n
    """
    if limit < 2:
        return [0] * (limit + 1)
    
    lpf = list(range(limit + 1))
    for i in range(2, isqrt(limit) + 1):
        if lpf[i] == i: # i is prime
            for j in range(i*i, limit + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    # Determine the maximum value we need to check LPF for
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Precompute LPF for all numbers up to max_val
    lpf = _compute_lpf(max_val)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Find any m in (lo, hi) such that lpf[m] >= gap
        # Note: If m is prime, lpf[m] = m. Since m > lo >= 10007 and gap <= 28, m >= gap is always true.
        # So primes in the gap are valid witnesses.
        # If m is composite, we need its smallest factor to be >= gap.
        
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
                
    # Ensure all entries are Python int or None
    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))
            
    return final_result
# EVOLVE-BLOCK-END
