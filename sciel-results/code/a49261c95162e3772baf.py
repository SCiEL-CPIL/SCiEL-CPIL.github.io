"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Generate powerful numbers up to M
    # M = 10^8 allows ~13000 numbers, ~8.5e7 pairs
    # This should run in ~10-20s
    M = 10**8
    
    powerful = set()
    max_b = int(M ** (1/3))
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(M // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= M:
                powerful.add(n)
    
    P = sorted(powerful)
    
    # Check time
    if time.time() > deadline - 30:
         # Fallback to smaller M if generation took too long (unlikely)
         # But generation is fast. The loop is the slow part.
         pass

    counts = {}
    n_len = len(P)
    
    # Count sums
    # We can break early if time is running out, but we need the best N.
    # If we break, we might miss the best N.
    # So we assume it finishes.
    
    for i in range(n_len):
        a = P[i]
        for j in range(i, n_len):
            b = P[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            
            # Optional: Time check inside loop?
            # if time.time() > deadline - 10: break
            # But breaking inner loop doesn't help much if we need consistent counts.
            # We will rely on M=10^8 being fast enough.
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}

    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Extract pairs
    pairs = []
    p_set = set(P)
    for a in P:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in p_set and a <= b:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
