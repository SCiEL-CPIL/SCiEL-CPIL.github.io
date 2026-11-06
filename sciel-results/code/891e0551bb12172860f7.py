import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s buffer before 150s kill
    
    # Phase 1: Generate powerful numbers up to L
    # Powerful number: every prime factor has exponent >= 2
    # Form: a^2 * b^3 where b is cube-free
    L = 10**9
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
    
    P = sorted(powerfuls)
    n_p = len(P)
    
    # Phase 2: Count pairwise sums using vectorized approach
    # For memory efficiency with large n_p, process in chunks
    # Total pairs: n_p*(n_p+1)/2
    
    import numpy as np
    
    P_arr = np.array(P, dtype=np.int64)
    
    # We need to count sums P[i] + P[j] for i <= j
    # Use chunked processing to avoid memory issues
    max_sum = 2 * L
    # bincount needs an array of size max_sum+1, which is 2*10^9 - too large
    # Instead, use a dict-based approach with vectorized chunks
    
    # Alternative: since max sum is 2*10^9, we can't use bincount directly
    # Use sorted approach: generate all sums, sort, count consecutive
    
    # For n_p ~ 13000, total pairs ~ 85M - too many to store
    # Use chunked approach: for each i, compute P[i] + P[i:] and count
    
    from collections import defaultdict
    counts = defaultdict(int)
    
    # Process in chunks to manage time
    chunk_size = 500  # process 500 rows at a time
    
    for i_start in range(0, n_p, chunk_size):
        if time.time() > deadline - 10:
            break
        i_end = min(i_start + chunk_size, n_p)
        
        # For rows i_start to i_end-1, compute sums with columns i to n_p-1
        # This is a triangular region
        for i in range(i_start, i_end):
            if time.time() > deadline - 10:
                break
            # P[i] + P[i:] as a vector
            sums_vec = P_arr[i:] + P_arr[i]
            # Count these sums
            # Use np.unique with return_counts for speed
            unique_sums, cnts = np.unique(sums_vec, return_counts=True)
            for s, c in zip(unique_sums, cnts):
                counts[int(s)] += int(c)
    
    # Phase 3: Find best N
    best_N = 0
    best_count = 0
    for s, c in counts.items():
        if c > best_count:
            best_count = c
            best_N = s
    
    # Phase 4: Construct pairs for best_N
    powerfuls_set = set(P)
    pairs = []
    for a in P:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
