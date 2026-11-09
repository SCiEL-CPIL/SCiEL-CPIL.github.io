# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    the minimum pairwise squared distance is >= the maximum squared norm.
    
    Uses all weight-2 vectors (exactly two non-zero entries, each +/-1), giving 220 points.
    Each has norm^2 = 2, and for any two, their dot product is in {-1, 0, 1},
    so ||p-q||^2 = 2 + 2 - 2*p.q >= 2 + 2 - 2*1 = 2 = max norm^2.
    """
    d = 11
    points_list = []
    
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
