# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using the E8 lattice roots construction to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    points_list = []
    
    # E8 roots construction:
    # Type A: (±1, ±1, 0, ..., 0) in 8 dimensions
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type B: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
    # Multiply by 2 to make integers: (±1, ±1, ..., ±1) with even number of -1s
    # In 8 dimensions, all 2^8 = 256 combinations, but only those with even number of -1s
    # Even number of minuses: 0, 2, 4, 6, 8
    # Count: C(8,0) + C(8,2) + C(8,4) + C(8,6) + C(8,8) = 1+28+70+28+1 = 128
    
    for mask in range(256):
        # Count number of set bits (minus signs)
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for k in range(8):
                if (mask >> k) & 1:
                    p[k] = -1
                else:
                    p[k] = 1
            points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the solution
    k = points.shape[0]
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Check no zero vector
    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        norms_sq = np.sum(points**2, axis=1)
    
    # Check pairwise distances using vectorized computation for speed
    min_dist_sq = float('inf')
    max_norm_sq = int(np.max(norms_sq))
    
    # Use numpy for efficient pairwise distance computation
    # But k can be up to 240, so k(k-1)/2 = ~28560 pairs, which is manageable
    # Let's do it in a loop but with early exit if we find dist_sq < max_norm_sq
    
    valid = True
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
        if time.time() > deadline:
            break
    
    # Check validity
    valid = valid and (min_dist_sq >= max_norm_sq)
    
    if not valid:
        # Fallback: use the D11 construction (220 points)
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        points = np.array(points_list, dtype=np.int64)
        k = points.shape[0]
        norms_sq = np.sum(points**2, axis=1)
        
        min_dist_sq = float('inf')
        max_norm_sq = int(np.max(norms_sq))
        valid = True
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
            # Final fallback: 2-point solution
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = np.around(points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
