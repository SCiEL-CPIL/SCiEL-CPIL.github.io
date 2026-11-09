import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance, 
    aiming to maximize the number of points.
    
    Strategy: Use all vectors with exactly two non-zero entries, each being +/- 1.
    Norm squared is 2. Minimum distance squared is 2.
    Condition min_dist^2 >= max_norm^2 becomes 2 >= 2, which holds.
    Number of points: C(11, 2) * 2^2 = 55 * 4 = 220.
    """
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
                
    points = np.array(points_list, dtype=np.int64)
    
    # Verification (optional but good for safety)
    # k = len(points_list)
    # if k < 2: return np.array([np.ones(d), -1*np.ones(d)], dtype=np.int64)
    
    # Check time
    if time.time() > deadline - 10:
        return points
    
    return points
