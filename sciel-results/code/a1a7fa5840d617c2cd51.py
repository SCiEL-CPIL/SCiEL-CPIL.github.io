import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all integer vectors of squared norm 2.
    There are C(11,2) * 2^2 = 55 * 4 = 220 such vectors.
    For any distinct pair u, v in this set, u . v <= 1.
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u . v = 4 - 2 u . v >= 4 - 2(1) = 2.
    Max squared norm is 2.
    Min squared distance is 2.
    Condition: 2 >= 2 holds.
    """
    d = 11
    points = []
    
    # Generate all vectors of norm squared 2
    # These are vectors with exactly two non-zero entries, each +/- 1
    for i in range(d):
        for j in range(i + 1, d):
            for sign_i in [1, -1]:
                for sign_j in [1, -1]:
                    v = [0] * d
                    v[i] = sign_i
                    v[j] = sign_j
                    points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array
