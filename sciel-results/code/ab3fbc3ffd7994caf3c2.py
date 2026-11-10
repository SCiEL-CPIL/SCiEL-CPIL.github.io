import numpy as np
from itertools import product, combinations

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate the 220 minimal vectors of the D11 lattice
    # These are vectors of the form +/- e_i +/- e_j for i != j
    # Norm squared is 1 + 1 = 2
    
    candidates = []
    for i, j in combinations(range(d), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * d
                v[i] = s1
                v[j] = s2
                candidates.append(tuple(v))
                
    # All these vectors are mutually compatible:
    # For any distinct u, v in this set, u dot v is in {-2, -1, 0, 1}.
    # Specifically, max dot product is 1.
    # Squared distance = ||u||^2 + ||v||^2 - 2(u dot v) = 2 + 2 - 2(1) = 2.
    # Max squared norm is 2.
    # Condition: min_sq_dist >= max_sq_norm => 2 >= 2. True.
    
    points_arr = np.array(candidates, dtype=np.int64)
    return points_arr
