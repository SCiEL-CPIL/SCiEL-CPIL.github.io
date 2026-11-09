import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Type A: Vectors with one coordinate +/- 2, rest 0. Norm squared = 4.
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            points.append(v)
            
    # Type B: Vectors with four coordinates +/- 1, rest 0. Norm squared = 4.
    # Choose 4 indices out of 11
    for indices in itertools.combinations(range(d), 4):
        i, j, k, l = indices
        # All 16 sign combinations
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        points.append(v)
                        
    arr = np.array(points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    
    return arr
