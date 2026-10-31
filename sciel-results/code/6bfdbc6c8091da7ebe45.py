import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    
    # Generate all vectors with exactly 4 non-zero entries, each +/- 1
    # Norm squared = 4, Norm = 2
    # Min distance between distinct such vectors is 2 (if they share 3 indices and differ in sign of the 4th)
    # 2 >= 2, so condition holds.
    
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
