"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Build least prime factor array using a sieve up to the maximum value needed
    max_val = max(hi for _, hi in P['gaps'])
    # We need LPF for all numbers up to max_val
    lpf = list(range(max_val + 1))
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve: for each prime p, mark its multiples with lpf = p if not already set
    # We only need to go up to sqrt(max_val) for the sieve itself
    import math
    limit = int(math.isqrt(max_val)) + 1
    for p in range(2, limit + 1):
        if lpf[p] == p:  # p is prime
            for multiple in range(p * p, max_val + 1, p):
                if lpf[multiple] == multiple:
                    lpf[multiple] = p
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Search for m in (lo, hi) with least prime factor >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
    
    # Ensure types are correct: Python int, not numpy or bool
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
