"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    vectors = []
    # Generate all vectors with exactly three non-zero entries, each +/- 1
    # Norm is sqrt(3)
    for indices in itertools.combinations(range(D), 3):
        i, j, k = indices
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
