import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Generate all pairs of indices (i, j) with i < j
    for i in range(D):
        for j in range(i + 1, D):
            # Generate all 4 sign combinations for the two positions
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
