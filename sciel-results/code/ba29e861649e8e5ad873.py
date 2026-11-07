"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second deadline with safety margin
    
    # Determine the maximum value we need to sieve up to
    max_prime = max(hi for _, hi in P['gaps'])
    
    # Sieve for least prime factor using NumPy for speed
    lp = np.arange(max_prime + 1, dtype=np.int64)
    lp[0] = 0
    lp[1] = 1
    
    for i in range(2, int(max_prime**0.5) + 1):
        if lp[i] == i:  # i is prime
            for j in range(i*i, max_prime + 1, i):
                if lp[j] == j:
                    lp[j] = i
    
    # For each gap, find a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m strictly between lo and hi such that lp[m] >= gap
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            if lp[m] >= gap:
                result[i] = m
                break
        
        # Check time
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
