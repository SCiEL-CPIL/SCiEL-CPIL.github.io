# EVOLVE-BLOCK-START
import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    corresponding to the minimal vectors of the D11 lattice.
    
    The D11 lattice minimal vectors are permutations of (+/-1, +/-1, 0, ..., 0).
    There are 2 * C(11, 2) = 220 such vectors.
    Max squared norm is 2.
    Min squared distance between distinct vectors is 2.
    Condition: min_dist^2 >= max_norm^2 => 2 >= 2. Valid.
    """
    d = 11
    points = []
    
    # Generate all pairs of indices for the non-zero coordinates
    for i in range(d):
        for j in range(i + 1, d):
            # Generate all sign combinations for the two non-zero coordinates
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    # Convert to numpy array
    arr = np.array(points)
    
    return arr

# EVOLVE-BLOCK-END
