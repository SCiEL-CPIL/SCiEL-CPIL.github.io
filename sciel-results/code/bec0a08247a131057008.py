"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    # Precompute primes up to 100 to handle any gap size comfortably
    # Max gap in data is 28, but 100 is safe and fast
    limit = 100
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Bad primes are those strictly less than the gap size
        # If m is not divisible by any prime p < gap, then lp(m) >= gap
        bad_primes = [p for p in primes if p < gap]
        
        found = False
        # Search for a witness in (lo, hi)
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            valid = True
            for p in bad_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
                
    return result
# EVOLVE-BLOCK-END
