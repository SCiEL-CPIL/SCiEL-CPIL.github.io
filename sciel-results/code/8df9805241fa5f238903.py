import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Limit 5 * 10^9
    # K ~ 70,000
    # K^2 ~ 5 * 10^9
    # Chunk size 5000 -> 14 chunks
    # Each chunk 3.5 * 10^8 elements
    # Estimated time 70-80s
    limit = 5000000000
    
    # Generate powerful numbers
    pows = set([1])
    max_i = math.isqrt(limit)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > limit:
            break
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
        if time.time() > deadline - 90: # Leave 90s for counting
             break

    P = np.array(sorted(list(pows)), dtype=np.int64)
    K = len(P)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Count sums using chunks
    # We count all ordered pairs (a, b)
    # Then distinct pairs a <= b is (count + delta) / 2
    counts = {}
    chunk_size = 5000
    
    for i_start in range(0, K, chunk_size):
        i_end = min(i_start + chunk_size, K)
        chunk = P[i_start:i_end]
        
        # Compute sums
        # chunk shape (C, 1), P shape (1, K)
        # Result (C, K)
        sums = chunk[:, None] + P[None, :]
        
        # Get unique sums and counts
        u_sums, u_counts = np.unique(sums, return_counts=True)
        
        # Update global counts
        # u_sums and u_counts are numpy arrays
        # Convert to python lists for faster dict update?
        # Or iterate directly
        for s, c in zip(u_sums, u_counts):
            counts[s] = counts.get(s, 0) + c
            
        if time.time() > deadline - 20:
            break

    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Find the sum with max count
    # Note: This count is for ordered pairs (a,b).
    # We want max number of distinct pairs a <= b.
    # M(S) = (C(S) + D(S)) / 2
    # We need to maximize M(S).
    # Since D(S) is 0 or 1, and C(S) is large, maximizing C(S) is a good proxy.
    # However, a sum with C(S)=100, D(S)=1 gives 50.5? No, C(S) must have same parity as D(S)?
    # C(S) = 2 * (pairs a<b) + (1 if a=b).
    # So C(S) and D(S) have same parity.
    # M(S) = (C(S) + D(S)) // 2.
    
    best_sum = 0
    best_score = 0
    
    pows_set = set(P.tolist())
    
    for s, c in counts.items():
        # s is numpy int64, convert to int for dict key consistency if needed, but keys are already int64 from numpy?
        # np.unique returns int64. dict keys will be int64.
        # pows_set has python ints.
        # Check if s/2 is in pows_set
        D = 0
        if s % 2 == 0:
            half = s // 2
            if half in pows_set:
                D = 1
        score = (c + D) // 2
        if score > best_score:
            best_score = score
            best_sum = s

    # Retrieve pairs for best_sum
    pairs = []
    for a in P:
        if a > best_sum:
            break
        b = best_sum - a
        if b < a:
            continue
        if b in pows_set:
            pairs.append([int(a), int(b)])
            
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': int(best_sum), 'pairs': pairs}
