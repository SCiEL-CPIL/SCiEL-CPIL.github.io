"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute least prime factor (lpf) for all numbers in the relevant range
_min_val = min(lo for lo, hi in P['gaps'])
_max_val = max(hi for lo, hi in P['gaps'])

def _compute_lpf(n):
    """Compute least prime factor for all numbers up to n using sieve."""
    lpf = list(range(n + 1))
    if n >= 0:
        lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

_lpf = _compute_lpf(_max_val)

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        # Check deadline
        if time.time() - _start_time > _DEADLINE:
            break
        
        gap = hi - lo
        
        # Find a witness m strictly between lo and hi
        # such that lpf[m] >= gap (least prime factor is at least the gap)
        # A number satisfies this if it has no prime factor < gap
        # i.e., it is either 1, prime >= gap, or product of primes all >= gap
        
        found = False
        for m in range(lo + 1, hi):
            if _lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
        if not found:
            # Try reverse search as well
            for m in range(hi - 1, lo, -1):
                if _lpf[m] >= gap:
                    result[i] = m
                    found = True
                    break
    
    # Ensure all entries are Python int or None
    return result

# EVOLVE-BLOCK-END
