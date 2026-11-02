import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Type 1: One 2, rest 0: ±2e_i (Norm sq 4)
    # 2 * 11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    # C(11, 4) * 2^4 = 330 * 16 = 5280 vectors
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vectors.append(v)
            
    # Total 5302 vectors
    # These are all mutually compatible with R^2 = 4.
    # Min distance squared = 4.
    # Max norm squared = 4.
    # Condition: ||u-v||^2 >= 4.
    # For u,v in this set, max dot product is 2.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2(2) = 4.
    # So distance is exactly 2 (norm), which satisfies >= 2.
    
    return np.array(vectors, dtype=np.int64)
