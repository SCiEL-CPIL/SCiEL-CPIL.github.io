"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Find the maximum number in the pairs to set sieve limit
    max_val = 0
    for lo, hi in P['gaps']:
        if hi > max_val:
            max_val = hi
    
    # Sieve for Least Prime Factor (LPF)
    # lpf[i] will store the smallest prime factor of i
    # For primes, lpf[p] = p
    # For 0 and 1, we don't care as we are looking for m > lo >= 10007
    lpf = list(range(max_val + 1))
    
    # Standard sieve modification for LPF
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        # Check deadline
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # We need m strictly between lo and hi
        # So m ranges from lo+1 to hi-1
        found = False
        for m in range(lo + 1, hi):
            # The condition is: least prime factor of m >= gap
            # lpf[m] is the least prime factor
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
        # If not found, result[i] remains None
    
    return result
# EVOLVE-BLOCK-END
