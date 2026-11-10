import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    min_n = 1000
    size = max_n + 25  # need sp up to 1000024

    # --- Precompute small-prime part (product of p^k for p<24) via NumPy sieve ---
    sp = np.ones(size, dtype=np.int64)
    for p in primes:
        p_power = p
        while p_power < size:
            sp[p_power - 1::p_power] *= p
            p_power *= p
        if time.time() > deadline - 40:
            break

    if time.time() > deadline - 30:
        return min_n

    # --- Rough prefilter: windows with the most sf==1 (fewest distinct candidates) ---
    rough = (sp == 1).astype(np.int32)
    cum_rough = np.zeros(size + 1, dtype=np.int32)
    np.cumsum(rough, out=cum_rough[1:])
    # window for n covers sp[n+1..n+24] -> indices n+1..n+24
    n_arr = np.arange(min_n, max_n + 1, dtype=np.int64)
    rough_counts = cum_rough[n_arr + 24] - cum_rough[n_arr]
    max_rough = int(np.max(rough_counts))
    candidates = n_arr[rough_counts == max_rough]

    # --- Map distinct sf values to bit indices for fast bitmask distinct-count ---
    vals, inv = np.unique(sp, return_inverse=True)
    nvals = len(vals)
    if nvals <= 60:
        bitmask = (1 << inv).astype(np.int64)
    else:
        bitmask = None

    best_val = float('inf')
    best_n = min_n

    # --- Pass 1: sliding window over the top-rough candidates ---
    if bitmask is not None:
        bm = bitmask
        # initialize window sum for n = min_n
        wsum = 0
        for i in range(1, 25):
            wsum |= bm[min_n + i]
        distinct = bin(wsum).count('1')
        if distinct < best_val:
            best_val = distinct
            best_n = min_n

        for n in candidates:
            if time.time() > deadline - 5:
                break
            n = int(n)
            if n == min_n:
                continue
            # slide from previous n to current n: window [n+1..n+24]
            # remove [prev+1..n], add [n+1..n+24] where prev is last processed n
            # Simpler: recompute incrementally using a running window
            # We'll maintain window by tracking last_n
            pass
        # Because candidates are sparse, recompute each window via bitmask OR of 24 entries
        # (cheap: 24 ORs each) for correctness
        for n in candidates:
            if time.time() > deadline - 5:
                break
            n = int(n)
            wsum = 0
            for i in range(1, 25):
                wsum |= bm[n + i]
            distinct = bin(wsum).count('1')
            if distinct < best_val:
                best_val = distinct
                best_n = n
                if best_val == 1:
                    break

    if best_val <= 2 or len(candidates) < 200:
        # already excellent or few candidates; try a quick full vectorized chunk polish
        pass

    # --- Pass 2: full vectorized chunk search (sort+diff distinct count) ---
    if time.time() < deadline - 8:
        chunk_size = 40000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 6:
                break
            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)
            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sp[indices]  # (chunk, 24)
            sorted_rows = np.sort(window_2d, axis=1)
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)
            min_idx = int(np.argmin(distinct_counts))
            min_count = int(distinct_counts[min_idx])
            if min_count < best_val:
                best_val = min_count
                best_n = int(chunk_n[min_idx])
                if best_val == 1:
                    break

    return int(best_n)
