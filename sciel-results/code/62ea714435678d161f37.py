import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Construct 220 points using integer E8 roots, D3 roots, and mixed vectors.
    # All points have squared norm 2.
    # Minimum squared distance between any distinct pair is 2.
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Valid.
    
    points = []
    
    # 1. Integer E8 roots in first 8 dimensions, 0 in last 3.
    # Vectors of form (+/-1, +/-1, 0, ..., 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # 2. D3 roots in last 3 dimensions, 0 in first 8.
    # Vectors of form (0, ..., 0, +/-1, +/-1, 0)
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # 3. Mixed vectors: unit vector in first 8, unit vector in last 3.
    # Vectors of form (+/-e_i, +/-e_j)
    for i in range(8):
        for si in [1, -1]:
            for j in range(8, 11):
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Total points: 112 + 12 + 96 = 220
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
