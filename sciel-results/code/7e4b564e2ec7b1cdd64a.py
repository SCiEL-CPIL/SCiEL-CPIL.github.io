# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Uses the minimal vectors of the D11 lattice.
    These are all vectors with exactly two non-zero coordinates, each being +1 or -1.
    Number of such vectors is C(11, 2) * 4 = 55 * 4 = 220.
    Max squared norm is 2.
    Min squared distance between distinct vectors is 2.
    Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Holds.
    """
    d = 11
    points = []
    
    # Generate all pairs of indices
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 sign combinations
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    vec = np.zeros(d, dtype=np.int64)
                    vec[i] = s1
                    vec[j] = s2
                    points.append(vec)
    
    # Convert to numpy array
    points_array = np.array(points, dtype=np.int64)
    
    return points_array


# EVOLVE-BLOCK-END
