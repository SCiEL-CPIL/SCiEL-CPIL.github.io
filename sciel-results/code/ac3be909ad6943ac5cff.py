import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Returns a set of 220 integer points in 11 dimensions.
    These points are all vectors with exactly two non-zero entries, each being +/- 1.
    This configuration satisfies the condition that the minimum squared distance
    between any two distinct points is >= the maximum squared norm of the points.
    """
    d = 11
    # Deadline guard, though this construction is O(1)
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    candidates = []
    # Generate all vectors with exactly two non-zero entries of magnitude 1
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # The set of 220 points is known to be valid and maximal for this constraint class
    points = np.array(candidates, dtype=np.int64)
    
    # Ensure interface requirements
    points = np.around(points).astype(np.int64)
    
    return points
