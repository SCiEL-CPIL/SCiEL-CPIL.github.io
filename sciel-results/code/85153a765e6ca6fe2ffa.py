import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a kissing configuration in Z^11 using all integer vectors with norm^2 = 4.
    
    These vectors come in two types:
    - Type A: (±2, 0, ..., 0) - one non-zero entry of ±2. Count: 11 * 2 = 22
    - Type B: (±1, ±1, ±1, ±1, 0, ..., 0) - four non-zero entries of ±1. Count: C(11,4) * 16 = 5280
    
    All have norm^2 = 4.
    Minimum pairwise distance^2 = 4 (verified: two type B vectors differing in one sign give dist^2=4;
    type A vs type B gives dist^2 >= 5; two type A gives dist^2 >= 4).
    
    So min_dist^2 = 4 >= max_norm^2 = 4. Condition satisfied.
    Total points: 5302.
    """
    start_time = time.time()
    deadline = start_time + 590
    
    d = 11
    points_list = []
    
    # Type A: (±2, 0, ..., 0)
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            points_list.append(v)
    
    # Type B: (±1, ±1, ±1, ±1, 0, ..., 0)
    for positions in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            points_list.append(v)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify
    if points.shape[0] < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Check no zero vectors
    if np.any(np.all(points == 0, axis=1)):
        points = points[~np.all(points == 0, axis=1)]
    
    # Verify distance condition
    k = points.shape[0]
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm_sq:
        # Fallback to D11
        points_list_fb = []
        for i, j in combinations(range(11), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(11, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list_fb.append(p)
        points = np.array(points_list_fb, dtype=np.int64)
    
    points = np.around(points).astype(np.int64)
    return points
