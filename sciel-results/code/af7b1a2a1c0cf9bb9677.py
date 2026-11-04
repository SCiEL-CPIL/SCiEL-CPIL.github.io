import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # The optimal known configuration for integer vectors in R^11
    # with min pairwise distance >= max norm is the set of all
    # vectors with exactly two non-zero entries, each +/-1.
    # These are the minimal vectors of the D_11 lattice.
    # Norm squared is 2. Min pairwise distance squared is 2.
    # Condition: 2 >= 2 holds.
    # Count: 2 * 11 * 10 = 220.
    
    vectors = []
    # Generate all pairs (i, j) with i < j
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations
            vectors.append([0] * D)
            v = vectors[-1]
            v[i] = 1; v[j] = 1
            
            vectors.append([0] * D)
            v = vectors[-1]
            v[i] = 1; v[j] = -1
            
            vectors.append([0] * D)
            v = vectors[-1]
            v[i] = -1; v[j] = 1
            
            vectors.append([0] * D)
            v = vectors[-1]
            v[i] = -1; v[j] = -1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return np.array(vectors, dtype=np.int64)
