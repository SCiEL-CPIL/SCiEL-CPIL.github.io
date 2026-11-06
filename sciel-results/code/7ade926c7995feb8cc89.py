import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 50.0

    min_n = 10000
    max_n = 1000000
    limit = max_n + 1  # we need LPF up to max_n + 1

    # Sieve for largest prime factor using NumPy vectorization
    lpf = np.zeros(limit + 1, dtype=np.int32)

    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            lpf[i::i] = i

    if time.time() > deadline:
        return min_n

    # M(n) = max(lpf[n], lpf[n+1]) for n in [min_n, max_n]
    # n ranges from min_n to max_n inclusive
    # lpf indices: lpf[n] for n in [min_n, max_n], lpf[n+1] for n in [min_n, max_n]
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]

    max_factors = np.maximum(left, right)
    min_idx = np.argmin(max_factors)
    best_n = min_n + min_idx

    return int(best_n)
