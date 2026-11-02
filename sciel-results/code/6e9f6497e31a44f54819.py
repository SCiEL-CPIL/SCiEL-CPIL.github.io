"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    vectors = []
    # Generate all vectors of the form +/- e_i +/- e_j
    # There are C(11, 2) = 55 pairs of indices.
    # For each pair, 4 sign combinations.
    # Total = 220 vectors.
    # Norm squared = 2. Norm = sqrt(2).
    # Min distance between distinct vectors = 2.
    # 2 >= sqrt(2) is True.
    
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
