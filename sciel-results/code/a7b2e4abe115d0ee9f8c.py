import time
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024

    # Precompute small prime part using vectorized sieve-like method
    small_part = np.ones(max_val + 1, dtype=np.int64)

    for p in primes:
        p_power = p
        while p_power <= max_val:
            indices = np.arange(p_power, max_val + 1, p_power)
            small_part[indices] *= p
            p_power *= p

    elapsed = time.time() - start_time
    if elapsed > 60:
        return 1000

    # Sliding window approach over n in [1000, 1000000]
    # Window for n is small_part[n+1 : n+25]
    best_n = 1000
    best_score = float('inf')

    # Initialize window for n=1000: indices 1001..1024
    window = small_part[1001:1025]
    wc = Counter(window.tolist())
    distinct_count = len(wc)
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000

    # Slide from n=1000 to n=999999
    # Transition from n to n+1: remove index n+1, add index n+25
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break

        # Remove element at index n (was n for previous window's start+1... let me re-derive)
        # Previous window (n-1): indices n, n+1, ..., n+23
        # Current window (n):   indices n+1, n+2, ..., n+24
        # So remove index n, add index n+24

        remove_val = int(small_part[n])
        add_val = int(small_part[n + 24])

        wc[remove_val] -= 1
        if wc[remove_val] == 0:
            del wc[remove_val]

        wc[add_val] += 1

        distinct_count = len(wc)
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break

    return best_n
