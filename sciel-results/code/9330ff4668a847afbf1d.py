"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Generate powerful numbers up to L
    # L=10^11 allows ~10,000 powerful numbers
    L = 10**11
    
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
                
    powerfuls_list = sorted(powerfuls)
    n_pows = len(powerfuls_list)
    
    # Use all powerful numbers if count is manageable for O(N^2) sum calculation
    # N=10,000 -> 100M sums. Memory 800MB. Time ~10s.
    # If N is larger, we might need to chunk or reduce L.
    if n_pows > 20000:
        # Fallback to smaller subset or lower L if it gets too big
        # For now, we assume L=10^11 is safe.
        pass

    P = np.array(powerfuls_list, dtype=np.int64)
    
    # Compute all pairwise sums
    # This creates an N x N matrix
    # To save memory, we can do it in chunks if N is very large, 
    # but 10k x 10k is 800MB, which is usually okay.
    # If MemoryError, we catch it and try a smaller chunk.
    
    try:
        sums = P[:, None] + P[None, :]
        sums_flat = sums.ravel()
    except MemoryError:
        # Fallback: Use a smaller subset of the smallest powerful numbers
        K = 5000
        P_chunk = P[:K]
        sums = P_chunk[:, None] + P_chunk[None, :]
        sums_flat = sums.ravel()
        
    # Find most frequent sums
    # unique is O(M log M)
    unique_sums, counts = np.unique(sums_flat, return_counts=True)
    
    # Find top 20 candidates
    # If counts is small, argpartition is fine.
    # If counts is large (up to 100M), argpartition is still O(M).
    top_indices = np.argpartition(counts, -20)[-20:]
    candidate_Ns = unique_sums[top_indices]
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_true_count = 1
    
    pows_set = set(powerfuls_list)
    
    for N in candidate_Ns:
        if time.time() > deadline - 10:
            break
            
        N = int(N)
        if N < 2:
            continue
            
        # Verify N with full set
        pairs = []
        for a in powerfuls_list:
            if 2 * a > N:
                break
            b_val = N - a
            if b_val in pows_set and b_val >= a:
                pairs.append([a, b_val])
                
        count = len(pairs)
        if count > best_true_count:
            best_true_count = count
            best_N = N
            best_pairs = pairs
            
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
