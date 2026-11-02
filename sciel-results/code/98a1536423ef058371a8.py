"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    vectors = []
    # Generate all vectors of the form +/- e_i +/- e_j
    # There are D*(D-1) such vectors
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            # 4 sign combinations
            vectors.append([0]*D)
            v = vectors[-1]
            v[i] = 1
            v[j] = 1
            
            vectors.append([0]*D)
            v = vectors[-1]
            v[i] = 1
            v[j] = -1
            
            vectors.append([0]*D)
            v = vectors[-1]
            v[i] = -1
            v[j] = 1
            
            vectors.append([0]*D)
            v = vectors[-1]
            v[i] = -1
            v[j] = -1
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
