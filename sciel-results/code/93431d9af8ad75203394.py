import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries, each +/- 1
    # There are C(11, 2) = 55 pairs of positions
    # For each pair, there are 4 sign combinations: (1,1), (1,-1), (-1,1), (-1,-1)
    # Total 55 * 4 = 220 vectors
    
    for pos in combinations(range(d), 2):
        for signs in product([-1, 1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[pos[0]] = signs[0]
            v[pos[1]] = signs[1]
            points.append(v)
            
    # Convert to numpy array
    # The shape will be (220, 11)
    pts_array = np.array(points, dtype=np.int64)
    
    # The task requires rounding and casting to int64, which is already done, 
    # but we follow the instruction strictly.
    return np.around(pts_array).astype(np.int64)
