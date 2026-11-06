import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 30s safety margin
    
    # Strategy:
    # 1. Generate powerful numbers up to L.
    # 2. Use direct pairwise sum matrix (int32) and np.bincount to find the N
    #    with the maximum number of representations.
    # 3. This avoids FFT overhead and memory issues of large float arrays.
    
    # L = 3 * 10^7
    # M ~ 2 * sqrt(L) ~ 11000
    # M^2 ~ 1.2 * 10^8
    # Memory for Sums (int32): 1.2e8 * 4 = 480 MB
    # Memory for counts (int64): ~6e7 * 8 = 480 MB
    # Total ~ 1 GB. Safe.
    
    L = 3 * 10**7
    
    # Generate powerful numbers
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    # print(f"Generated {M} powerful numbers up to {L} in {elapsed_gen:.1f}s")
    
    if time.time() > deadline - 20:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy int32 array
    P = np.array(powerfuls_list, dtype=np.int32)
    
    # Compute all pairwise sums
    # P[:, None] is Mx1, P[None, :] is 1xM
    # Result is MxM
    try:
        Sums = P[:, None] + P[None, :]
    except MemoryError:
        # If memory fails, reduce L and retry? Or just fallback.
        # For this proposal, we assume L=3e7 fits.
        return {'N': 2, 'pairs': [[1, 1]]}
    
    flat_sums = Sums.ravel()
    
    # Count occurrences of each sum
    # np.bincount requires non-negative integers
    counts = np.bincount(flat_sums)
    
    elapsed_count = time.time() - start_time
    # print(f"Counting done in {elapsed_count:.1f}s")
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with the maximum number of unordered pairs (a <= b)
    # Ordered count is counts[N].
    # Unordered count U(N):
    # If N is odd: counts[N] // 2
    # If N is even: (counts[N] + 1) // 2 if N//2 is powerful, else counts[N] // 2
    
    powerfuls_set = set(powerfuls_list)
    
    # Get top 100 indices with highest ordered counts
    # np.argpartition is O(N)
    k = min(100, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    
    best_N = 2
    best_unordered_count = 0
    
    for N in top_indices:
        N = int(N)
        if N < 2:
            continue
        ordered = int(counts[N])
        if N % 2 == 0 and (N // 2) in powerfuls_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        
        if unordered > best_unordered_count:
            best_unordered_count = unordered
            best_N = N
    
    # Extract pairs for best_N
    pairs = []
    # Binary search or set lookup
    # Since P is sorted, we can iterate
    # a <= b => a <= best_N // 2
    # b = best_N - a
    
    # Using set for O(1) lookup
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    elapsed_total = time.time() - start_time
    # print(f"Total time: {elapsed_total:.1f}s, Best N: {best_N}, Count: {best_unordered_count}")
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
