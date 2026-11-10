import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of all vectors with exactly two non-zero entries, each being +/- 1.
    This set has size 220 (the kissing number of the D_11 lattice).
    It satisfies the condition min_dist_sq >= max_norm_sq (both are 2).
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    d = 11
    vectors = []
    
    # Iterate over all pairs of indices (i, j)
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all 4 sign combinations for positions i and j
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    points = np.array(vectors, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points
