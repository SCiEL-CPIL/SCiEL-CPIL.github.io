"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Precompute least prime factors up to the maximum number in the gaps
    max_val = max(hi for lo, hi in P['gaps'])
    # Add a small buffer just in case, though max_val is the upper bound exclusive
    limit = max_val + 10
    lpf = list(range(limit))
    
    # Sieve of Eratosthenes to find least prime factor
    for i in range(2, isqrt(limit) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, limit, i):
                if lpf[j] == j:
                    lpf[j] = i

    result = [None] * len(P['gaps'])
    
    # 2. Find witnesses for each gap
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap_size = hi - lo
        # We need m such that lo < m < hi and lpf[m] >= gap_size
        # We scan all candidates in the gap to ensure we find one if it exists
        # Since gaps are small, this is very fast
        found = False
        # Optimization: Check candidates in a way that might find "large lpf" numbers sooner?
        # Actually, any valid witness is equally good for the score.
        # Just finding *any* one is sufficient.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                result[i] = m
                found = True
                break
        # If no witness found, result[i] remains None

    # Ensure types are correct (int or None)
    # lpf values are ints, indices are ints, so result[i] = m is int.
    return result

# EVOLVE-BLOCK-END
