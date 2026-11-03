import numpy as np
import itertools
import time

def construct():
    D = 11
    vectors = []
    
    # Generate all pairs of indices (supports of size 2)
    for support in itertools.combinations(range(D), 2):
        i, j = support
        # Generate all 4 sign combinations for the two non-zero entries
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                v[i] = s1
                v[j] = s2
                vectors.append(v)
                
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    return final_vectors
