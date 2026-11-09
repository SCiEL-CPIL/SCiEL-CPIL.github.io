import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    # This gives C(11,2) * 4 = 55 * 4 = 220 points
    # All have norm^2 = 2, and min pairwise distance^2 = 4 >= 2
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    points = np.array(points_list, dtype=np.int64)
    
    # Verify basic validity
    if points.shape[1] != d:
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
        
    k = points.shape[0]
    if k < 2:
        return points
        
    # Check no zero vector (shouldn't happen with our construction)
    if np.any(np.all(points == 0, axis=1)):
        mask = np.any(points != 0, axis=1)
        points = points[mask]
        k = points.shape[0]
        
    # All points have norm^2 = 2, so max_norm_sq = 2
    # Min pairwise distance^2 is 4 (verified analytically), so valid
    # Quick sanity check with a subset of pairs to avoid full O(k^2) if needed
    max_norm_sq = 2  # All points have exactly two +/-1 entries
    
    # Verify with a sample of pairs for safety
    min_dist_sq = float('inf')
    # Check all pairs - 220 points gives ~24k pairs, fast enough
    for i in range(k):
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff * diff))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Should not happen; fallback to A11 roots
                fallback_list = []
                for i2, j2 in combinations(range(d), 2):
                    p = np.zeros(d, dtype=np.int64)
                    p[i2] = 1
                    p[j2] = -1
                    fallback_list.append(p)
                return np.array(fallback_list, dtype=np.int64)
                
    return np.around(points).astype(np.int64)
