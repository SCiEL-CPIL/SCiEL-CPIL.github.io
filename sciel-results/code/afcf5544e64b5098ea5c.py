"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute primes globally to avoid re-computation if construct is called multiple times
_MAX_VAL = max(hi for _, hi in P['gaps'])
_LIMIT = int(math.isqrt(_MAX_VAL)) + 2

def _generate_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

_GLOBAL_PRIMES = _generate_primes(_LIMIT)

def construct():
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        found = False
        for m in range(lo + 1, hi):
            lpf = m
            for p in _GLOBAL_PRIMES:
                if p * p > m:
                    break
                if m % p == 0:
                    lpf = p
                    break
            
            if lpf >= gap:
                result[i] = int(m)
                found = True
                break
                
    return result
# EVOLVE-BLOCK-END
