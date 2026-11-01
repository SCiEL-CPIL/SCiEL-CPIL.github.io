import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    vectors = []
    # Generate all integer vectors with exactly two non-zero entries, each +/- 1
    # These have norm sqrt(2).
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                if time.time() > deadline:
                    break
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
        
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
         # Return best found so far
         if vectors:
             return np.array(vectors, dtype=np.int64)
         else:
             # Fallback to a small valid set
             fallback = []
             for i in range(D):
                 v = [0]*D
                 v[i] = 1
                 fallback.append(v)
                 v[i] = -1
                 fallback.append(v)
             return np.array(fallback, dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
