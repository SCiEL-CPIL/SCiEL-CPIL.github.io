"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute small primes up to 20 (max gap in dataset is 16, so primes < 16 are needed)
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]

def construct():
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need LPF(m) >= gap.
        # This is equivalent to: m is not divisible by any prime p < gap.
        
        # Find relevant primes < gap
        forbidden_primes = [p for p in _SMALL_PRIMES if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any forbidden prime
            is_valid = True
            for p in forbidden_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
            
    return result
# EVOLVE-BLOCK-END
