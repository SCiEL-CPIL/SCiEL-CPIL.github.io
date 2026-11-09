"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    # lp[m] = smallest prime p < 24 dividing m, or 0 if none
    lp = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for p in PRIMES:
        # Only set lp where it hasn't been set yet (i.e., no smaller prime divides m)
        indices = np.arange(p, MAX_VAL + 1, p)
        mask = lp[indices] == 0
        if mask.any():
            lp[indices[mask]] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    # sp[m] = product of all prime powers p^k where p < 24 and p^k divides m.
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p != 0:
            sp[m] = sp[m // p] * p

    # Sliding window with frequency tracking for O(1) distinct count updates
    cnt = Counter()
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1, WINDOW + 1):
        v = int(sp[1000 + i])
        cnt[v] += 1
        if cnt[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
        # Remove sp[n] (was the leftmost element of previous window [n, n+23])
        # Add sp[n+24] (rightmost element of new window [n+1, n+24])
        v_remove = int(sp[n])
        cnt[v_remove] -= 1
        if cnt[v_remove] == 0:
            del cnt[v_remove]
            distinct -= 1

        v_add = int(sp[n + WINDOW])
        old_add = cnt.get(v_add, 0)
        cnt[v_add] = old_add + 1
        if old_add == 0:
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
