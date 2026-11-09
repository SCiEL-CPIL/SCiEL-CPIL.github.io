import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    n_min = 1000
    n_max = 1000000

    # The numbers we are factoring are n+1, n+2, ..., n+24.
    # So the range of values is [n_min+1, n_max+24]
    val_min = n_min + 1
    val_max = n_max + 24

    # Precompute G[m] = product of p^v_p(m) for all p < 24
    # G array corresponds to values n_min+1 to n_max+24
    # Index i in G corresponds to value (n_min + 1 + i)

    indices = np.arange(val_min, val_max + 1, dtype=np.int64)
    G = np.ones(len(indices), dtype=np.int64)

    for p in primes:
        p_power = p
        while p_power <= val_max:
            mask = (indices % p_power == 0)
            G[mask] *= p
            p_power *= p
            if time.time() > deadline:
                return 1000

    # Use vectorized sliding window distinct count
    # Create windows of size 24 using stride tricks
    # G has length size + 23
    # Number of windows = size
    # windows[i] = G[i:i+24]

    windows = np.lib.stride_tricks.sliding_window_view(G, 24)
    # windows has shape (size, 24)

    # Sort each row to count distinct values
    sorted_windows = np.sort(windows, axis=1)

    # Count distinct: number of positions where sorted_windows[i,j] != sorted_windows[i,j-1]
    # Plus 1 for the first element
    diffs = np.diff(sorted_windows, axis=1)  # shape (size, 23)
    distinct_counts = (diffs != 0).sum(axis=1) + 1

    # Find the minimum
    min_idx = np.argmin(distinct_counts)
    best_n = n_min + min_idx

    return int(best_n)
