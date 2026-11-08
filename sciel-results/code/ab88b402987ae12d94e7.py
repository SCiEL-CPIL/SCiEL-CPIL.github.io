import itertools
import numpy as np

def construct():
    D = 11
    vectors = []
    
    # Generate all vectors with exactly 4 non-zero entries (+/- 1)
    # There are C(11, 4) ways to choose the positions
    # and 2^4 ways to choose the signs
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
