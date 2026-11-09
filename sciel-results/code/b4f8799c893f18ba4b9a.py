import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    points = np.array(points_list)
    
    # Verify validity
    if points.shape[1] != d:
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
        
    k = points.shape[0]
    if k < 2:
        return points
        
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        # Filter out zero vectors if any (shouldn't happen)
        mask = np.any(points != 0, axis=1)
        points = points[mask]
        k = points.shape[0]
        
    # Check min dist sq >= max norm sq
    max_norm_sq = np.max(np.sum(points**2, axis=1))
    
    min_dist_sq = float('inf')
    # Loop over pairs
    for i in range(k):
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, fallback to smaller set
                # Fallback: roots of A11 (e_i - e_j)
                fallback_list = []
                for i2, j2 in combinations(range(d), 2):
                    p = np.zeros(d, dtype=np.int64)
                    p[i2] = 1
                    p[j2] = -1
                    fallback_list.append(p)
                return np.array(fallback_list).astype(np.int64)
                
    return points
