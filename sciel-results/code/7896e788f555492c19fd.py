import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all integer vectors with exactly two non-zero entries, each +/- 1
    # These vectors have norm squared = 2.
    # There are 4 such vectors for each pair of indices (i, j) with i < j.
    # Total count: 4 * C(11, 2) = 4 * 55 = 220.
    
    points = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # Vector 1: e_i + e_j
            p1 = np.zeros(d, dtype=np.int64)
            p1[i] = 1
            p1[j] = 1
            points.append(p1)
            
            # Vector 2: e_i - e_j
            p2 = np.zeros(d, dtype=np.int64)
            p2[i] = 1
            p2[j] = -1
            points.append(p2)
            
            # Vector 3: -e_i + e_j
            p3 = np.zeros(d, dtype=np.int64)
            p3[i] = -1
            p3[j] = 1
            points.append(p3)
            
            # Vector 4: -e_i - e_j
            p4 = np.zeros(d, dtype=np.int64)
            p4[i] = -1
            p4[j] = -1
            points.append(p4)
            
    points_array = np.array(points, dtype=np.int64)
    
    # Verification (optional but good for safety)
    if len(points_array) >= 2:
        max_norm_sq = np.max(np.sum(points_array ** 2, axis=1))
        min_dist_sq = float('inf')
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                diff = points_array[i] - points_array[j]
                dsq = np.dot(diff, diff)
                if dsq < min_dist_sq:
                    min_dist_sq = dsq
        if min_dist_sq < max_norm_sq:
            # Fallback to a smaller valid set if something unexpected happened
            # Just return the first 2 points
            return points_array[:2]
            
    return points_array
