# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points corresponding to
    the roots of the D11 lattice (all vectors with exactly two non-zero
    entries of +/- 1). This yields 220 points where the minimum squared
    distance between distinct points is 4, while the maximum squared norm
    is 2, satisfying the constraint that min_dist_sq >= max_norm_sq.
    """
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    # Convert to numpy array and round/cast as required
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points

# EVOLVE-BLOCK-END
