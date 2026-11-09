import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    The set consists of all vectors in Z^11 with squared norm 2.
    There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    The minimum squared distance between any two distinct vectors is 2.
    The maximum squared norm is 2.
    Condition: min_dist^2 >= max_norm^2 => 2 >= 2. Valid.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations for indices i and j
            # 1, 1
            p = np.zeros(d, dtype=np.int64)
            p[i] = 1
            p[j] = 1
            points.append(p)
            
            # 1, -1
            p = np.zeros(d, dtype=np.int64)
            p[i] = 1
            p[j] = -1
            points.append(p)
            
            # -1, 1
            p = np.zeros(d, dtype=np.int64)
            p[i] = -1
            p[j] = 1
            points.append(p)
            
            # -1, -1
            p = np.zeros(d, dtype=np.int64)
            p[i] = -1
            p[j] = -1
            points.append(p)
            
    return np.array(points)
