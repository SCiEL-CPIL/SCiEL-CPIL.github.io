import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m] = product of p^k for p < 24 where p^k | m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Use sliding window view
    start_idx = N_MIN + 1
    num_windows = N_MAX - N_MIN + 1
    data = extracted[start_idx:start_idx + num_windows + WINDOW - 1]
    windows = np.lib.stride_tricks.sliding_window_view(data, WINDOW)

    # Sort each window and count distinct
    sorted_windows = np.sort(windows, axis=1)
    distinct_counts = 1 + np.sum(sorted_windows[:, 1:] != sorted_windows[:, :-1], axis=1)

    min_val = int(np.min(distinct_counts))
    min_idx = int(np.argmin(distinct_counts))
    best_n = int(N_MIN + min_idx)

    return best_n
