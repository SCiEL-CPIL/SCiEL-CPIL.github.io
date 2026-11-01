"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # Generate all weight-2 vectors with entries in {-1, 1} and 0 elsewhere.
    # Norm squared is 2.
    # Condition for this set: min pairwise distance >= max norm.
    # Max norm = sqrt(2).
    # Min distance between distinct vectors is sqrt(2) (e.g., (1,1,0..) and (1,-1,0..)).
    # sqrt(2) >= sqrt(2) holds.
    # Thus, the set of all such vectors is a valid kissing configuration.
    
    vectors = []
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
