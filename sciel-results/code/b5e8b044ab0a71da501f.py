import time
import numpy as np
from numpy.lib.stride_tricks import as_strided


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW  # 1000024

    # Compute sp[m] = product of p^v_p(m) for all primes p < 24, for m in 1..MAX_VAL
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in PRIMES:
        # All multiples of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, compute v_p(m) = exponent of p in m
        # m = p * k, so v_p(m) = 1 + v_p(k)
        # Use iterative division in a loop (fast in NumPy for vectorized ops)
        temp = multiples.copy()
        exp = np.zeros(len(multiples), dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            exp[mask] += 1
            temp[mask] //= p
        # Multiply sp[multiples] by p^exp
        sp[multiples] *= np.power(p, exp, dtype=np.int64)

    # Now scan for n in [1000, N] minimizing distinct count of sp[n+1..n+24]
    best_n = 1000
    best_distinct = WINDOW + 1  # worst case: all 24 distinct

    chunk_size = 50000

    for chunk_start in range(1000, N + 1, chunk_size):
        if time.time() > deadline - 5:
            break

        chunk_end = min(chunk_start + chunk_size, N + 1)
        num_windows = chunk_end - chunk_start

        # base array: sp[chunk_start+1 : chunk_start + num_windows + WINDOW]
        base = sp[chunk_start + 1 : chunk_start + num_windows + WINDOW]
        row_stride = base.strides[0]
        mat = as_strided(base, shape=(num_windows, WINDOW), strides=(row_stride, row_stride))

        # Sort each row and count distinct via diffs
        sorted_mat = np.sort(mat, axis=1)
        diffs = np.diff(sorted_mat, axis=1)
        distinct_counts = (diffs != 0).sum(axis=1) + 1

        local_min_idx = int(np.argmin(distinct_counts))
        local_min_val = int(distinct_counts[local_min_idx])

        if local_min_val < best_distinct:
            best_distinct = local_min_val
            best_n = chunk_start + local_min_idx
            if best_distinct == 1:
                break

    return best_n
