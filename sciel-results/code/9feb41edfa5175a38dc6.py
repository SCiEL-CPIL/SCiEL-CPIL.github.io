import numpy as np
import time
import itertools

D = 11

def construct():
    """
    Constructs a set of 242 integer vectors in R^11.
    The vectors are of the form:
    1. +/- 2 * e_i  (11 * 2 = 22 vectors)
    2. +/- 2 * e_i +/- 2 * e_j (55 * 4 = 220 vectors)
    Total: 242 vectors.
    Max norm squared R^2 = 8.
    Min distance squared is >= 8.
    """
    vectors = []
    
    # Type 1: +/- 2 * e_i
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
            
    # Type 2: +/- 2 * e_i +/- 2 * e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
