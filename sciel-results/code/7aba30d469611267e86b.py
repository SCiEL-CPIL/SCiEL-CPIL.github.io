import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = 110.0

    def elapsed():
        return time.time() - start_time

    D = 11
    
    # Generate all weight-2 vectors (norm sqrt(2))
    # These form the D_11 lattice minimal vectors
    vecs = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    vecs.append(v)
    
    vecs = np.array(vecs, dtype=np.int64)
    
    # This set of 220 vectors is valid.
    # Min dist sqrt(2), max norm sqrt(2).
    # Condition: sqrt(2) >= sqrt(2). Holds.
    
    # Try to improve by adding more?
    # For now, return this robust solution.
    
    return vecs
