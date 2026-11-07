"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s out of 90s limit, safety margin
    
    # The full range of numbers we need to check
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    
    import math
    
    # Standard sieve for LPF up to max_val
    # lpf[n] will be the least prime factor of n, or n if n is prime
    lpf = list(range(max_val + 1))
    
    limit = int(math.isqrt(max_val)) + 1
    
    for p in range(2, limit + 1):
        if lpf[p] == p:  # p is prime
            # Mark multiples of p starting from p*p
            for m in range(p * p, max_val + 1, p):
                if lpf[m] == m:  # not yet marked by a smaller prime
                    lpf[m] = p
    
    # Now for each gap, find a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        # Scan m from lo+1 to hi-1
        for m in range(lo + 1, hi):
            factor = lpf[m]
            if factor >= gap_size:
                result[i] = m
                break
        # Check deadline
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
