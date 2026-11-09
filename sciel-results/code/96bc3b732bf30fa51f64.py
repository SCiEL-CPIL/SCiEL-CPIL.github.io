import numpy as np
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate 2E8 vectors (8 dimensions)
    # Type 1: +/- 2 e_i +/- 2 e_j
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    points.append(np.concatenate([v, np.zeros(3, dtype=np.int64)]))
                    
    # Type 2: +/- e_1 ... +/- e_8 with even number of minus signs
    # This corresponds to 2 * (1/2)(+/- e_1 ... +/- e_8)
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            points.append(np.concatenate([v, np.zeros(3, dtype=np.int64)]))
            
    # Generate 2D3 vectors (3 dimensions)
    # 2D3 minimal vectors are +/- 2 e_i +/- 2 e_j
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    points.append(np.concatenate([np.zeros(8, dtype=np.int64), v]))
                    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
