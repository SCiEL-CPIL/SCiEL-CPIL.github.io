# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs 220 integer lattice points in R^11 with the property that
    the minimum squared pairwise distance is >= the maximum squared norm.
    
    Uses all vectors of the form (+/-1, +/-1, 0, ..., 0) with exactly two
    nonzero coordinates. Count: C(11,2) * 4 = 55 * 4 = 220.
    All have norm^2 = 2.
    Minimum pairwise distance^2 is 2 (vectors sharing one coordinate with
    same sign, differing in the other). So min_dist_sq = 2 >= max_norm_sq = 2.
    """
    d = 11
    points_list = []
    
    # Generate all weight-2 vectors: exactly two coordinates are +/-1, rest are 0
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
                    
    points = np.array(points_list, dtype=np.int64)
    
    # Quick verification
    assert points.shape == (220, 11)
    norms_sq = np.sum(points * points, axis=1)
    assert np.all(norms_sq == 2)
    
    return points

# EVOLVE-BLOCK-END
