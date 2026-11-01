"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate D11 shortest vectors (norm squared 2, integer coords)
    # These are vectors with two coordinates +/-1 and rest 0.
    # Sum of coordinates must be even.
    # (+1, +1) sum 2 (even)
    # (-1, -1) sum -2 (even)
    # (+1, -1) sum 0 (even)
    # (-1, +1) sum 0 (even)
    # So all combinations are valid.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
                    
    result = np.array(vectors, dtype=np.int64)
    
    return result
