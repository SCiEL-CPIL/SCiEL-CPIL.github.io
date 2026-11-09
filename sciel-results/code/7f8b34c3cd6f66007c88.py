# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the
    minimum pairwise squared distance is >= the maximum squared norm.
    
    Construction: Use the direct sum structure 2*E8 ⊕ 2*D3.
    - 2*E8 has 240 minimal vectors of norm^2 = 8 in 8 dimensions.
    - 2*D3 has 12 minimal vectors of norm^2 = 8 in 3 dimensions.
    
    We take the union of:
    1. Points from 2*E8 padded with 0 in the last 3 dims (240 points, norm^2=8)
    2. Points from 2*D3 padded with 0 in the first 8 dims (12 points, norm^2=8)
    
    Total points: 252.
    Max norm squared: 8.
    Min pairwise squared distance: 8 (verified for this union structure).
    Condition: min_dist_sq >= max_norm_sq holds (8 >= 8).
    """
    deadline = time.time() + 540  # safety margin
    
    d = 11
    
    # --- Generate 2*E8 vectors (240 points) in first 8 dimensions ---
    # E8 minimal vectors:
    # 112 weight-2: ±e_i ± e_j, i<j. Scaled by 2: ±2e_i ± 2e_j. Norm^2 = 4+4=8.
    # 128 weight-8: ±1/2 in all 8 coords, even number of minus signs. Scaled by 2: ±1 in all 8 coords. Norm^2 = 8.
    
    e8_vectors = []
    
    # Weight-2 vectors: ±2e_i ± 2e_j for 0 <= i < j < 8
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
    
    # Weight-8 vectors: ±1 in all 8 coordinates, with even number of minus signs
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            e8_vectors.append(list(signs))
    
    # Pad E8 to 11D (last 3 dims are 0)
    e8_points = []
    for v in e8_vectors:
        vec = v + [0, 0, 0]
        e8_points.append(vec)
    
    # --- Generate 2*D3 vectors (12 points) in last 3 dimensions ---
    # D3 minimal vectors: ±e_i ± e_j for i<j in 3D. Scaled by 2: ±2e_i ± 2e_j. Norm^2 = 8.
    d3_vectors = []
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 3
                v[i] = 2 * si
                v[j] = 2 * sj
                d3_vectors.append(v)
    
    # Pad D3 to 11D (first 8 dims are 0)
    d3_points = []
    for v in d3_vectors:
        vec = [0] * 8 + v
        d3_points.append(vec)
    
    # Combine all points
    all_points = e8_points + d3_points
    
    points_arr = np.array(all_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    return points_arr

# EVOLVE-BLOCK-END
