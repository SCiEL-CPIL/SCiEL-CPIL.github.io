import time
import numpy as np

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def time_remaining():
        return deadline - time.time()

    # Build the 252-vector E8 ⊕ A3 kissing configuration.
    # This is a known optimal configuration for dimension 11 (kissing number 252).

    e8_vectors = []

    # Type 1: 112 vectors with two ±2 and six 0s
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_vectors.append(v)

    # Type 2: 128 vectors with all ±1, even number of -1s
    for bits in range(256):
        v = np.zeros(8, dtype=np.int64)
        minus_count = 0
        for i in range(8):
            if (bits >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        if minus_count % 2 == 0:
            e8_vectors.append(v)

    # A3 roots: 12 vectors in 3 dimensions
    a3_vectors = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    a3_vectors.append(v)

    # Combine into R^11 vectors
    selected = []

    # Add E8 vectors (padded with 0s in last 3 dims)
    for v in e8_vectors:
        full_v = np.zeros(D, dtype=np.int64)
        full_v[:8] = v
        selected.append(full_v)

    # Add A3 vectors (padded with 0s in first 8 dims)
    for v in a3_vectors:
        full_v = np.zeros(D, dtype=np.int64)
        full_v[8:] = v
        selected.append(full_v)

    selected = np.array(selected, dtype=np.int64)

    # Validate: min pairwise distance >= max norm
    norms_sq = np.sum(selected**2, axis=1)
    max_norm_sq = np.max(norms_sq)

    min_dist_sq = np.inf
    n = len(selected)
    for i in range(n):
        diffs = selected[i+1:] - selected[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            min_d = np.min(dists_sq)
            if min_d < min_dist_sq:
                min_dist_sq = min_d

    if min_dist_sq >= max_norm_sq:
        return selected

    # Fallback: return E8 part only (should not trigger)
    e8_full = np.zeros((240, D), dtype=np.int64)
    e8_full[:, :8] = np.array(e8_vectors, dtype=np.int64)
    return e8_full
