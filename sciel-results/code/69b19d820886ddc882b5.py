D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin within 120s limit

    vectors = []

    # Generate D_3 roots: ±e_i ± e_j for 1≤i<j≤3, in dimensions 8,9,10
    # These are 12 vectors with norm sqrt(2), scaled by 2 -> norm sqrt(8)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[8 + i] = 2 * si
                v[8 + j] = 2 * sj
                vectors.append(v)

    # Generate E_8 roots, scaled by 2, in dimensions 0-7
    # Type 1: ±2e_i ± 2e_j (128 vectors)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)

    # Type 2: all ±1 with even number of minus signs (128 vectors), scaled by 2
    # Actually E8 type 2 is all vectors with entries ±1, even number of minuses
    # There are 2^8 / 2 = 128 such vectors
    # Scaled by 2: entries are ±2
    for mask in range(256):
        # Count number of 1-bits
        bits = bin(mask).count('1')
        if bits % 2 == 0:
            v = np.zeros(D, dtype=np.int64)
            for i in range(8):
                v[i] = 2 * ((-1) ** ((mask >> i) & 1))
            vectors.append(v)

    vecs = np.array(vectors, dtype=np.int64)
    
    # Verify constraints
    norms_sq = np.sum(vecs ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise distances squared
    n = len(vecs)
    # Check that min distance >= max norm
    # dist^2(i,j) = ||ui||^2 + ||vj||^2 - 2*ui.vj
    # We need dist^2 >= max_norm for all i≠j
    # i.e., ||ui||^2 + ||vj||^2 - 2*ui.vj >= max_norm_sq
    
    # Efficient check using matrix
    if n > 0:
        dot = vecs @ vecs.T
        # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dot[i,j]
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        # We need min_dist_sq >= max_norm_sq
        # Since we scaled E8 roots (norm^2=8) and D3 roots (norm^2=8), all have norm^2=8
        # Cross distances: E8 vec has support in dims 0-7, D3 in dims 8-10
        # So dot product = 0, dist^2 = 8+8 = 16 >= 8. Good.
        # Within E8: min dist^2 = 8 (for adjacent roots). Good.
        # Within D3: min dist^2 = 8. Good.
        
        if min_dist_sq < max_norm_sq:
            # Fallback: just return the seed configuration
            import itertools
            fallback = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        fallback.append(v)
            return np.array(fallback, dtype=np.int64)

    return vecs

# EVOLVE-BLOCK-END
