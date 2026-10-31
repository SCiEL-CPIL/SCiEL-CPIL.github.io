import itertools
import numpy as np

def construct():
    D = 11
    vectors = []
    # Generate all binary vectors with exactly 4 non-zero entries (+/- 1)
    # Indices of non-zero entries
    for support in itertools.combinations(range(D), 4):
        # Signs for the 4 entries
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
