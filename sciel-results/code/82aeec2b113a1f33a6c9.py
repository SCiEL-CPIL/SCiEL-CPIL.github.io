"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    vectors = []
    # Use vectors with exactly 4 non-zero entries, each +/- 1
    # Norm squared is 4, so norm is 2.
    # Min distance squared is 4, so min distance is 2.
    # Condition: min_dist >= max_norm => 2 >= 2. Holds.
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
