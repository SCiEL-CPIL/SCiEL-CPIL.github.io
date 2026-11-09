import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all weight-2 vectors in Z^11
    # A weight-2 vector has exactly two non-zero entries, which are +/- 1.
    # Number of such vectors: C(11, 2) * 2^2 = 55 * 4 = 220.
    
    points = []
    for combo in combinations(range(d), 2):
        for signs in product([-1, 1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[combo[0]] = signs[0]
            v[combo[1]] = signs[1]
            points.append(v)
            
    result = np.array(points, dtype=np.int64)
    return result
