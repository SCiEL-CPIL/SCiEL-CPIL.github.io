"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    vectors = []
    # Generate all weight-2 vectors in R^11
    # These are the minimal vectors of the D_11 lattice
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations: (1,1), (1,-1), (-1,1), (-1,-1)
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    # Check time, though generation is instant
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)
        
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
