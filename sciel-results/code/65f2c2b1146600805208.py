# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using the scaled E8 lattice roots and scaled D3 lattice roots.
    
    E8 roots (scaled by 2):
    - Type A: (±2, ±2, 0, ..., 0) in 8 dimensions. 112 points.
    - Type B: (±1, ±1, ..., ±1) in 8 dimensions with even number of -1s. 128 points.
    
    D3 roots (scaled by 2):
    - (0, ..., 0, ±2, ±2, 0) in last 3 dimensions. 12 points.
    
    Total: 112 + 128 + 12 = 252 points.
    All points have norm squared 8.
    Minimum squared distance is 8.
    Condition: min_dist_sq >= max_norm_sq => 8 >= 8. Valid.
    """
    d = 11
    points_list = []
    
    # E8 Type A: (±2, ±2, 0, ..., 0)
    for i, j in combinations(range(8), 2):
        for si in [2, -2]:
            for sj in [2, -2]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    # E8 Type B: (±1, ±1, ..., ±1) with even number of -1s
    for mask in range(256):
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for k in range(8):
                if (mask >> k) & 1:
                    p[k] = -1
                else:
                    p[k] = 1
            points_list.append(p)
            
    # D3 Type: (0, ..., 0, ±2, ±2, 0) in last 3 dims (indices 8, 9, 10)
    # Indices: 8, 9, 10
    # Pairs: (8,9), (8,10), (9,10)
    for i, j in combinations([8, 9, 10], 2):
        for si in [2, -2]:
            for sj in [2, -2]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    points = np.array(points_list, dtype=np.int64)
    
    # Verification
    k = points.shape[0]
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
    
    # Check pairwise distances
    # k=252, pairs ~ 31k. Loop is fast.
    for i in range(k):
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
            
    if not valid or min_dist_sq < max_norm_sq:
        # Fallback to D11 roots (220 points)
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        points = np.array(points_list, dtype=np.int64)
        
    return points

# EVOLVE-BLOCK-END
