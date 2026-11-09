import time
import numpy as np
from numpy.lib.stride_tricks import as_strided


def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW  # 1000024

    # Compute sp[m] = product of p^v_p(m) for all primes p < 24, for m in 1..MAX_VAL
    # Faster sieve-like approach
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in PRIMES:
        # Multiply by p for all multiples of p
        sp[p::p] *= p
        # Multiply by p for all multiples of p^2, p^3, etc.
        power = p * p
        while power <= MAX_VAL:
            sp[power::power] *= p
            power *= p

    # Scan for n in [1000, N] minimizing distinct count of sp[n+1..n+24]
    best_n = 1000
    best_distinct = WINDOW + 1  # worst case: all 24 distinct

    # Vectorized scan in chunks to manage memory and time
    # We process windows starting at n from 1000 to N
    # The values are sp[n+1] to sp[n+WINDOW]
    
    chunk_size = 50000
    start_n = 1000
    end_n = N + 1
    
    for chunk_start in range(start_n, end_n, chunk_size):
        if time.time() > deadline - 5:
            break
            
        chunk_end = min(chunk_start + chunk_size, end_n)
        num_windows = chunk_end - chunk_start
        
        # We need sp[chunk_start + 1] to sp[chunk_end - 1 + WINDOW]
        # Index range: [chunk_start + 1, chunk_end + WINDOW - 1]
        base = sp[chunk_start + 1 : chunk_end + WINDOW]
        
        # Create view of shape (num_windows, WINDOW)
        # The stride between consecutive windows is 1 element in sp
        row_stride = base.strides[0]
        mat = as_strided(base, shape=(num_windows, WINDOW), strides=(row_stride, row_stride))
        
        # Sort each row
        sorted_mat = np.sort(mat, axis=1)
        # Count distincts: 1 + number of non-zero diffs
        diffs = np.diff(sorted_mat, axis=1)
        distinct_counts = (diffs != 0).sum(axis=1) + 1
        
        # Find min in this chunk
        local_min_idx = int(np.argmin(distinct_counts))
        local_min_val = int(distinct_counts[local_min_idx])
        
        if local_min_val < best_distinct:
            best_distinct = local_min_val
            best_n = chunk_start + local_min_idx
            if best_distinct <= 1:
                break

    # Optional: Refine around best_n if time permits and best_distinct is not 1
    # Given the vectorized scan is quite fast, a refinement might not be necessary
    # unless the global minimum is rare. But let's do a small polish just in case.
    if time.time() < deadline - 2 and best_distinct > 1:
        ref_start = max(1000, best_n - 100)
        ref_end = min(N, best_n + 100)
        for n in range(ref_start, ref_end + 1):
            if time.time() > deadline - 1:
                break
            window = sp[n + 1 : n + 1 + WINDOW]
            sorted_w = np.sort(window)
            diffs = np.diff(sorted_w)
            dc = int((diffs != 0).sum()) + 1
            if dc < best_distinct:
                best_distinct = dc
                best_n = n

    return best_n
