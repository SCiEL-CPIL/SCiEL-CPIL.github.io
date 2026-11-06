import time
from math import isqrt
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Generate powerful numbers up to L
    L = 10**8
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L: break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = (a * a) * b3
            if val <= L:
                powerfuls.add(val)
    
    P = np.array(sorted(powerfuls), dtype=np.int64)
    n_p = len(P)
    
    # 2. Count sums using batched 2D NumPy operations
    # For a block of i values, create sums[i,j] = P[i] + P[j] for j >= i
    best_N = 0
    best_count = 0
    counts = defaultdict(int)
    
    # Process in blocks to manage memory
    block_size = 2000
    
    for i_start in range(0, n_p, block_size):
        if time.time() > deadline - 15:
            break
        
        i_end = min(i_start + block_size, n_p)
        
        # For each i in [i_start, i_end), compute sums with P[i:]
        # We need to handle the triangular structure
        # Process one row at a time but use efficient NumPy
        for i in range(i_start, i_end):
            a = P[i]
            sums = a + P[i:]
            unique_sums, cnts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, cnts):
                counts[s] = counts.get(s, 0) + int(c)
                if counts[s] > best_count:
                    best_count = counts[s]
                    best_N = s
    
    # 3. Construct result
    powerfuls_set = set(P)
    pairs = []
    for a in P:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
