"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Determine the maximum value to sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Precompute Least Prime Factor (LPF) for all numbers up to max_val
    # lpf[i] will be the smallest prime factor of i
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Check if we are running out of time
        if time.time() > deadline:
            break
            
        # Search for a witness m in (lo, hi)
        # Condition: lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
            # Optional: early termination if time is critical, but the loop is very fast
            if time.time() > deadline:
                break

    return result
# EVOLVE-BLOCK-END
