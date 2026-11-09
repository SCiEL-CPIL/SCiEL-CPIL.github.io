# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all vectors of the form +/- e_i +/- e_j for i != j.
    There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    All have squared norm 2.
    Minimum squared distance between distinct vectors is 2.
    Condition: min_dist^2 >= max_norm^2 => 2 >= 2. Valid.
    """
    d = 11
    points = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 combinations of signs for e_i and e_j
            # +e_i +e_j
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            
            # +e_i -e_j
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points.append(v2)
            
            # -e_i +e_j
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            points.append(v3)
            
            # -e_i -e_j
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            points.append(v4)
            
    return np.array(points)

# EVOLVE-BLOCK-END
