import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    # Generate all vectors of the form (+/-1, +/-1, 0, ..., 0)
    # There are C(11, 2) pairs of indices
    # For each pair, 4 sign combinations
    for i in range(d):
        for j in range(i + 1, d):
            # Signs for i and j
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Convert to array
    points = np.array(points, dtype=np.int64)
    return points
