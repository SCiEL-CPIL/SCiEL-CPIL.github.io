"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    result = [None] * len(P['gaps'])
    
    # Determine the range of numbers to check
    min_val = min(lo for lo, _ in P['gaps'])
    max_val = max(hi for _, hi in P['gaps'])
    
    # Precompute least prime factor (LPF) for all numbers in [min_val, max_val]
    # We only need to check up to max_val.
    # LPF[n] will store the least prime factor of n.
    # If n is prime, LPF[n] = n.
    # We can use a sieve-like approach.
    
    lpf = [0] * (max_val + 1)
    # Initialize lpf for all numbers to 0 (unknown)
    # Sieve for primes up to max_val
    # For each prime p, mark multiples of p with LPF=p if not already set.
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    # Now lpf[m] gives the least prime factor for any m in range.
    # For primes, lpf[m] = m.
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
