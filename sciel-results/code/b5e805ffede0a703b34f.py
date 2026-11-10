"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    N = 123457
    
    # Precompute largest prime factor for all integers up to N
    # Using a sieve-like approach in O(N log log N)
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # i is the largest prime factor seen so far for j
    
    # Handle lpf[1] = 1 as per problem statement
    lpf[1] = 1
    
    # Find the split a + b = N that minimizes max(P(a), P(b))
    best_max_pf = float('inf')
    best_a = 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        m = max(lpf[a], lpf[b])
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
