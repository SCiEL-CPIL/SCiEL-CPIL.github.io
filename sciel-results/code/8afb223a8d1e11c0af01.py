"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # We need to evaluate n in [1000, 1000000], so we look at n+i for i in 1..24
    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    lp = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for p in PRIMES:
        lp[p::p] = np.where(lp[p::p] == 0, p, lp[p::p])

    # Compute sp[m] = complete factor of m composed of primes < 24
    # Use iterative approach with numpy for speed
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    # For each m, sp[m] = sp[m // lp[m]] * lp[m] if lp[m] != 0, else 1
    # Process in order from 1 to MAX_VAL
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p != 0:
            sp[m] = sp[m // p] * p

    # Sliding window over n in [1000, N], window is sp[n+1], sp[n+2], ..., sp[n+24]
    # For each window, count distinct values, minimize count
    best_n = 1000
    best_distinct = float('inf')

    # Use sliding window with Counter for O(1) updates
    # Initial window: n=1000, values sp[1001] to sp[1024]
    window_vals = sp[1001:1025]  # sp[n+1] to sp[n+24] for n=1000
    cnt = Counter(window_vals.tolist())
    distinct_count = len(cnt)
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000

    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        # Remove sp[n] (which was sp[(n-1)+1]) and add sp[n+24]
        old_val = sp[n]
        new_val = sp[n + 24]
        cnt[old_val] -= 1
        if cnt[old_val] == 0:
            del cnt[old_val]
            distinct_count -= 1
        cnt[new_val] += 1
        if new_val not in cnt or cnt[new_val] == 1:
            distinct_count += 1
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
