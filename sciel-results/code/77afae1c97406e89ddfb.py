import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate the 220 vectors with two +/-1 entries (norm squared 2)
    candidates = []
    for support in combinations(range(d), 2):
        for signs in product([1, -1], repeat=2):
            v = [0] * d
            v[support[0]] = signs[0]
            v[support[1]] = signs[1]
            candidates.append(tuple(v))
            
    # All distinct pairs in this set have dot product <= 1.
    # Norm squared is 2 for all.
    # Min squared distance is 4 - 2*1 = 2.
    # Max squared norm is 2.
    # Condition: 2 >= 2 holds.
    # Thus, the entire set of 220 vectors is a valid configuration.
    
    points_arr = np.array(candidates, dtype=np.int64)
    return points_arr
