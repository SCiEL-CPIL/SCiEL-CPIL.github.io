import itertools
import numpy as np

def construct():
    D = 11
    vectors = []
    # Generate all combinations of 4 indices
    for indices in itertools.combinations(range(D), 4):
        # Generate all sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
