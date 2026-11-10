import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the roots of the D_11 lattice: all vectors of the form (+-1, ..., +-1, ..., 0, ...)
    with exactly two non-zero entries.
    
    There are 11*10*2 = 220 such vectors.
    Norm squared is 2.
    Minimum squared distance between distinct roots is 2.
    Condition: min_dist_sq (2) >= max_norm_sq (2). Holds.
    """
    d = 11
    points = []
    
    # Generate all pairs i, j with i < j
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations for e_i and e_j
            # (+1, +1), (+1, -1), (-1, +1), (-1, -1)
            # Vector v1: v1[i]=1, v1[j]=1
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = 1
            points.append(v.copy())
            
            # Vector v2: v2[i]=1, v2[j]=-1
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            points.append(v.copy())
            
            # Vector v3: v3[i]=-1, v3[j]=1
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = 1
            points.append(v.copy())
            
            # Vector v4: v4[i]=-1, v4[j]=-1
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = -1
            points.append(v.copy())
            
    arr = np.array(points, dtype=np.int64)
    return arr
