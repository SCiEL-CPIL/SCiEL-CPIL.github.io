"""Kissing number in dimension 11: construct nonzero integer vectors with min pairwise distance >= max norm."""
import time
import numpy as np

D = 11
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import itertools
import random
import copy
import sys

def construct():
    start_time = time.time()
    
    # Strategy: Return the set of all weight-2 integer vectors.
    # There are 220 such vectors in R^11.
    # They all have norm^2 = 2.
    # The minimum pairwise distance between any two distinct weight-2 vectors is sqrt(2).
    # Proof:
    #   Let u, v be distinct weight-2 vectors.
    #   ||u||^2 = 2, ||v||^2 = 2.
    #   ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 - 2<u,v>.
    #   <u,v> is an integer.
    #   Possible values for <u,v>:
    #     If u and v share 2 indices with same signs: <u,v>=2. But then u=v. Distinct implies not this.
    #     If u and v share 2 indices with opposite signs: <u,v>=-2. ||u-v||^2 = 4 - 2(-2) = 8.
    #     If u and v share 1 index: <u,v> is 1 or -1. ||u-v||^2 = 4 - 2(1) = 2 or 4 - 2(-1) = 6.
    #     If u and v share 0 indices: <u,v>=0. ||u-v||^2 = 4.
    #   So min ||u-v||^2 is 2.
    #   Max norm is sqrt(2).
    #   Constraint: min dist >= max norm -> sqrt(2) >= sqrt(2). True.
    #   Thus, this set of 220 vectors is a valid kissing configuration.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Ensure shape is (k, D)
    if result.size == 0:
        return np.empty((0, D), dtype=np.int64)
        
    return result
# EVOLVE-BLOCK-END
