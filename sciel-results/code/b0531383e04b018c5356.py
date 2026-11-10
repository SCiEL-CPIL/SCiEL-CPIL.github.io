import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of all vectors with exactly two non-zero entries (+/- 1).
    This yields 220 points.
    """
    d = 11
    points = []
    
    # Generate all pairs of indices
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 sign combinations for indices i and j
            # (1, 1), (1, -1), (-1, 1), (-1, -1)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points.append(p)
    
    # Convert to array
    points_array = np.array(points)
    
    return points_array
