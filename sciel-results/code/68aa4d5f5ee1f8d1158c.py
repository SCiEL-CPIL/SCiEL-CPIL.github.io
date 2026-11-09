import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    points_list = []
    
    # All 220 vectors of type (+/-1, +/-1, 0, ...) in Z^11
    # These have norm^2 = 2, and min pairwise dist^2 = 2
    for i, j in combinations(range(11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    points = np.array(points_list, dtype=np.int64)
    n = len(points)
    
    # Verification
    norms = np.sum(points**2, axis=1)
    assert np.all(norms == 2), f"Not all norms are 2: {np.unique(norms)}"
    assert n >= 2, "Need at least 2 points"
    assert np.all(norms > 0), "No point should be zero"
    
    min_dist_sq = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                if min_dist_sq < 2:
                    break
        if min_dist_sq < 2:
            break
        if time.time() > deadline:
            break
            
    assert min_dist_sq >= 2, f"Min dist^2 = {min_dist_sq} < 2"
    
    return points
