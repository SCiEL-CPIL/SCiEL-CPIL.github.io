"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    # lp[m] = smallest prime < 24 dividing m, or 0 if none
    lp = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for p in reversed(PRIMES):
        lp[p::p] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    m = np.arange(0, MAX_VAL + 1, dtype=np.int64)
    
    # Iteratively divide out small prime factors
    for _ in range(20):
        p_div = lp[m]
        mask = p_div > 0
        if not np.any(mask):
            break
        sp[mask] *= p_div[mask]
        m[mask] //= p_div[mask]

    # Convert to list for faster Python access in sliding window? 
    # Or keep as numpy array and access. Numpy array access is slower than list in Python loops?
    # Actually, list of ints is faster for individual access in a tight Python loop than numpy scalar access.
    sp_list = sp.tolist()

    # Sliding window with frequency tracking for O(1) distinct count updates
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1, WINDOW + 1):
        v = sp_list[1000 + i]
        freq[v] += 1
        if freq[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        # Remove sp[n], add sp[n+24]
        v_remove = sp_list[n]
        freq[v_remove] -= 1
        if freq[v_remove] == 0:
            distinct -= 1

        v_add = sp_list[n + WINDOW]
        freq[v_add] += 1
        if freq[v_add] == 1:
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
