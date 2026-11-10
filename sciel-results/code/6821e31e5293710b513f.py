# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm squared
    is less than or equal to their minimum pairwise squared distance, maximizing the number of points.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    # Generate all vectors of the form (e_i + e_j), (e_i - e_j), (-e_i + e_j), (-e_i - e_j) for i != j
    # This covers all integer vectors with squared norm 2 (two non-zero entries, each +/-1)
    points = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Add the zero vector
    points.append(np.zeros(d, dtype=np.int64))
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
