"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Find the range we need to sieve
    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])
    
    # Sieve for least prime factor up to all_hi
    size = all_hi + 1
    lpf = [0] * size  # lpf[i] = least prime factor of i
    for i in range(size):
        lpf[i] = i
    
    # Sieve of Eratosthenes to fill lpf
    for i in range(2, isqrt(all_hi) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, size, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Search for m in (lo, hi) such that lpf[m] >= gap
        # lpf[m] >= gap means m has no prime factor < gap
        # For m itself, lpf[m] = m if m is prime, else smallest prime divisor
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)  # Ensure Python int
                break
    
    return result
# EVOLVE-BLOCK-END
