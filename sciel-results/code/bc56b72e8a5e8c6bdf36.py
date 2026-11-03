import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    # Generate all combinations of 4 indices
    for indices in itertools.combinations(range(D), 4):
        # Generate all sign patterns for these 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
