"""Sieve-based sliding window search for minimum distinct smooth factors."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 1000000
    W = 24  # window size

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Precompute smooth part for every number in [1, N]
    # smooth_part[m] = product of prime powers p^a for p < 24 where p^a || m
    smooth_part = np.ones(N + 1, dtype=np.int32)
    for p in primes:
        for m in range(p, N + 1, p):
            v = m
            while v % p == 0:
                smooth_part[m] *= p
                v //= p

    # Slide window: for each n in [1000, N-W], window is smooth_part[n+1 : n+1+W]
    # Use np.unique on the slice to count distinct values
    best_n = 1000
    best_count = W

    # Use dict-based sliding window with minimal overhead
    freq = {}
    sp = smooth_part  # local reference

    # Initialize window for n = 1000: indices 1001..1024
    for i in range(1001, 1001 + W):
        val = sp[i]
        freq[val] = freq.get(val, 0) + 1

    distinct = len(freq)
    best_n = 1000
    best_count = distinct

    if best_count <= 3:
        return best_n

    # Slide the window
    check_interval = 200000
    next_check = check_interval

    for n in range(1001, N - W + 1):
        # Remove index n (leftmost of previous window: window was [n, n+W-1])
        sp_remove = sp[n]
        cnt = freq[sp_remove] - 1
        if cnt == 0:
            del freq[sp_remove]
        else:
            freq[sp_remove] = cnt

        # Add index n+W
        sp_add = sp[n + W]
        freq[sp_add] = freq.get(sp_add, 0) + 1

        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 3:
                break

        if n >= next_check:
            if time.time() > deadline:
                break
            next_check += check_interval

    return best_n
