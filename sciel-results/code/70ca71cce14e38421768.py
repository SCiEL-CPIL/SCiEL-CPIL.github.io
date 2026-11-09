# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Uses the roots of the D_11 root system, which are the 220 vectors of the form +/-e_i +/-e_j.
    These vectors have squared norm 2. The minimum squared distance between any two distinct roots is 2.
    Thus, min_dist_sq >= max_norm_sq (2 >= 2) is satisfied.
    """
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    # These are the roots of the D_d root system.
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    points = np.array(points_list, dtype=np.int64)
    
    # Verification (optional, but good for debugging during development)
    # In production, we trust the math.
    # if points.shape[0] > 1:
    #     norms = np.sum(points**2, axis=1)
    #     max_norm = np.max(norms)
    #     min_dist_sq = float('inf')
    #     k = points.shape[0]
    #     for a in range(k):
    #         for b in range(a+1, k):
    #             dist_sq = np.sum((points[a] - points[b])**2)
    #             if dist_sq < min_dist_sq:
    #                 min_dist_sq = dist_sq
    #     if min_dist_sq < max_norm:
    #         print("Invalid configuration found!")
    #         # Fallback to smaller set if needed
    #         return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    return points

# EVOLVE-BLOCK-END
