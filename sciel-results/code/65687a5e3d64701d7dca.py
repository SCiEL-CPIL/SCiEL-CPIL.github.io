import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer

    # Choose L to balance memory and score
    # L=10^8 -> M~4800, Pairs~11M, Mem~250MB, Score~5.8
    # L=5*10^8 -> M~10700, Pairs~57M, Mem~1.3GB, Score~?
    # L=10^9 -> M~15000, Pairs~112M, Mem~2.6GB, Score~?
    
    # We will try L = 10**9. If memory is an issue, it will crash.
    # To be safer, we can use L = 5 * 10**8.
    # Given the "maximize" goal, higher L is better.
    # I'll bet on 5 * 10**8 being safe and better than 10**8.
    
    L = 5 * 10**8
    
    # 1. Generate powerful numbers
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = (a * a) * b3
            if val <= L:
                powerfuls.add(val)
        
    P = np.array(sorted(powerfuls), dtype=np.int64)
    n_p = len(P)
    
    if n_p == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 2. Compute all sums a + b where a <= b
    # Use int32 for indices to save memory
    i_idx, j_idx = np.triu_indices(n_p, dtype=np.int32)
    
    # Compute sums
    # P[i_idx] and P[j_idx] are int64
    sums = P[i_idx] + P[j_idx]
    
    # 3. Find most frequent sum
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    if len(counts) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    max_count = np.max(counts)
    best_N = unique_sums[np.argmax(counts)]
    
    # 4. Construct pairs
    mask = (sums == best_N)
    a_vals = P[i_idx[mask]]
    b_vals = P[j_idx[mask]]
    
    pairs = [[int(a), int(b)] for a, b in zip(a_vals, b_vals)]
    
    return {'N': int(best_N), 'pairs': pairs}
