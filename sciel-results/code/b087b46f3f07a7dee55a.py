"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Group A: 220 weight-2 vectors scaled by 2 (two entries ±2, rest 0), norm^2 = 8
    group_A = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                group_A.append(v)
    
    # Group B: 240 E8 half-integer vectors scaled by 2 (all 8 first-dim entries ±1, last 3 dims 0), norm^2 = 8
    # E8 minimal half-integer vectors: all coordinates ±1/2 with even number of minus signs
    # Scaled by 2: all coordinates ±1 with even number of minus signs
    group_B = []
    for signs in itertools.product([1, -1], repeat=8):
        # Count minus signs
        minus_count = sum(1 for s in signs if s == -1)
        if minus_count % 2 == 0:
            v = list(signs) + [0, 0, 0]
            group_B.append(v)
    
    # Combine
    all_vectors = group_A + group_B
    
    # Verify constraints
    vecs = np.array(all_vectors, dtype=np.int64)
    n = len(vecs)
    
    # Check all norms are equal
    norms_sq = np.sum(vecs ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Check minimum pairwise distance
    # dist^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # For efficiency, compute Gram matrix
    if n > 5000:
        # Sample check
        idx = np.random.choice(n, size=min(n, 5000), replace=False)
        sample = vecs[idx]
        gram = sample @ sample.T
        dist_sq = norms_sq[idx][:, None] + norms_sq[idx][None, :] - 2 * gram
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
    else:
        gram = vecs @ vecs.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * gram
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
    
    # Check validity
    if min_dist_sq < max_norm_sq:
        # Fallback: just return Group A
        return np.array(group_A, dtype=np.int64)
    
    return vecs
# EVOLVE-BLOCK-END
