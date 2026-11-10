"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 123457

    # Largest prime factor sieve: for each prime p (detected as lpf[p]==0),
    # stamp p onto every multiple. Since we iterate p in increasing order,
    # the final value stored in lpf[x] is the largest prime dividing x.
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    lpf[1] = 1

    best_max_pf = float('inf')
    best_a = 1

    # Scan all splits a + b = N, using O(1) lookups.
    # We only need to check a in [1, N//2] since the pair (a,b) is symmetric.
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max_pf:
            best_max_pf = m
            best_a = a

    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
