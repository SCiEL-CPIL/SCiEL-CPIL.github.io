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
    
    # We need LPF for numbers in [min_val+1, max_val-1]
    # Let's create an array indexed by n - min_val
    offset = min_val
    size = max_val - min_val
    
    # Initialize LPF array: lpf[n] = least prime factor of n, or n if n is prime
    lpf = list(range(size))
    
    # Sieve: for each prime p, mark multiples
    import math
    limit = int(math.isqrt(max_val)) + 1
    
    for p in range(2, limit + 1):
        if lpf[p - offset] == p:  # p is prime
            # Mark multiples of p starting from p*p
            start = p * p
            if start >= min_val:
                for m in range(start, max_val, p):
                    idx = m - offset
                    if idx >= 0 and idx < size:
                        if lpf[idx] == m:  # not yet marked
                            lpf[idx] = p
    
    # Now for each gap, find a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        # Scan m from lo+1 to hi-1
        for m in range(lo + 1, hi):
            idx = m - offset
            if idx < 0 or idx >= size:
                continue
            factor = lpf[idx]
            if factor >= gap_size:
                result[i] = m
                break
        # Check deadline
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
