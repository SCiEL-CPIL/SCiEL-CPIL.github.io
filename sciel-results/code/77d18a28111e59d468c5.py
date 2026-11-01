import numpy as np
import itertools
import time

def construct():
    """
    Returns a list of integer vectors in R^11 forming a kissing configuration.
    The vectors are weight-2 vectors (entries in {-1, 0, 1} with exactly two non-zero entries).
    There are 220 such vectors, and they satisfy the condition that the minimum pairwise
    distance is at least the maximum norm (which is sqrt(2)).
    """
    D = 11
    
    # Generate all weight-2 vectors
    # v[i] = s1, v[j] = s2, where s1, s2 in {1, -1}
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[i] = s1
                v[j] = s2
                vectors.append(v)
    
    # Convert to numpy array
    vecs = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify validity (for debugging/safety, though we know it's valid)
    # Norm squared is 2 for all.
    # Max norm is sqrt(2).
    # Distance squared between distinct vectors:
    # If they share 1 index: 1+1-2*dot. Dot is 1 or -1.
    # If dot=1, dist^2 = 2+2-2 = 2 >= 2. OK.
    # If dot=-1, dist^2 = 2+2-(-2) = 6 >= 2. OK.
    # If they share 0 indices: 2+2-0 = 4 >= 2. OK.
    # If they share 2 indices (same support):
    # Distinct means signs differ in at least one pos.
    # If 1 sign differs: dot = 1-1=0. dist^2 = 4. OK.
    # If 2 signs differ: dot = -1-1=-2. dist^2 = 8. OK.
    
    return vecs
