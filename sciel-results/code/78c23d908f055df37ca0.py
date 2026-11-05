import numpy as np
import time
from numpy.lib.stride_tricks import as_strided

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW  # need smooth parts up to 1000024

    # Precompute 23-smooth parts: smooth[m] = product of p^k for p<24 where p^k || m
    # Using power-based slicing: for each p and each power p^j, multiply all multiples of p^j by p.
    smooth = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in primes:
        power = p
        while power <= MAX_VAL:
            smooth[power::power] *= p
            power *= p

    best_n = N_MIN
    best_count = WINDOW + 1

    total_windows = N_MAX - N_MIN + 1
    CHUNK = 50000

    for chunk_start in range(0, total_windows, CHUNK):
        if time.time() > deadline:
            break
        chunk_end = min(chunk_start + CHUNK, total_windows)
        n_start = N_MIN + chunk_start
        chunk_len = chunk_end - chunk_start

        # Windows for n in [n_start, n_start+chunk_len):
        # window for n uses smooth[n+1 : n+1+WINDOW]
        # flat range needed: smooth[n_start+1 : n_start+chunk_len+WINDOW]
        flat = smooth[n_start + 1 : n_start + chunk_len + WINDOW]

        # Build a (chunk_len, WINDOW) view via stride trick
        row_stride = flat.strides[0]
        windows_2d = as_strided(
            flat,
            shape=(chunk_len, WINDOW),
            strides=(row_stride, row_stride)
        )

        # Count distinct per row: sort, then count value-changes
        sorted_w = np.sort(windows_2d, axis=1)
        changes = np.diff(sorted_w, axis=1)
        distinct_counts = 1 + np.sum(changes != 0, axis=1)

        local_min_idx = int(np.argmin(distinct_counts))
        local_min_val = int(distinct_counts[local_min_idx])

        if local_min_val < best_count:
            best_count = local_min_val
            best_n = n_start + local_min_idx
            if best_count == 1:
                break

    return best_n
