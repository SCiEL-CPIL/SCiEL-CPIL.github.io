import numpy as np
from itertools import combinations, product
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the direct sum of 2*E8 roots and 2*D3 roots to maximize the number of points.
    
    E8 roots (in 8D) scaled by 2: 240 points, norm^2 = 8.
    D3 roots (in 3D) scaled by 2: 12 points, norm^2 = 8.
    Combined in 11D: 252 points.
    Min dist^2 = 8, Max norm^2 = 8. Valid.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for safety

    points_list = []

    # 1. Generate E8 roots in the first 8 dimensions
    # Type A: (±2, ±2, 0, ..., 0)
    # C(8,2) * 4 = 112 roots
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type B: (±2, ±2, ±2, ±2, ±2, ±2, ±2, ±2) with even number of minuses
    # 128 roots
    for mask in range(256):
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for k in range(8):
                if (mask >> k) & 1:
                    p[k] = -2
                else:
                    p[k] = 2
            points_list.append(p)

    # 2. Generate D3 roots in the last 3 dimensions (indices 8, 9, 10)
    # D3 roots are vectors with two non-zero entries (±1).
    # Scaled by 2: entries are ±2.
    # Positions: (8,9), (8,10), (9,10)
    tail_indices = [(8, 9), (8, 10), (9, 10)]
    for i, j in tail_indices:
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)

    points = np.array(points_list, dtype=np.int64)
    k = points.shape[0]
    
    # Verification
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        norms_sq = np.sum(points**2, axis=1)
        
    max_norm_sq = int(np.max(norms_sq))
    min_dist_sq = float('inf')
    valid = True
    
    # Efficient check
    # k=252, pairs ~ 31k. Very fast.
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    if valid and min_dist_sq >= max_norm_sq:
        return np.around(points).astype(np.int64)
    else:
        # Fallback to D11 if something went wrong (should not happen)
        points_list_fb = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list_fb.append(p)
        points_fb = np.array(points_list_fb, dtype=np.int64)
        return np.around(points_fb).astype(np.int64)

# EVOLVE-BLOCK-END
