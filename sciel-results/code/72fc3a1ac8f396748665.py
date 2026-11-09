import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # E8 lattice minimal vectors (scaled by 2) in the first 8 dimensions
    # Type A: Two coordinates are ±2, rest 0 in the first 8 dims
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [-2, 2]:
                for sj in [-2, 2]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Type B: All 8 coordinates are ±1, with even number of -1s
    # This gives 128 vectors
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.zeros(d, dtype=np.int64)
            for i in range(8):
                v[i] = signs[i]
            points.append(v)
    
    # D3 lattice minimal vectors (scaled by 2) in the last 3 dimensions
    # These are vectors with two non-zero coordinates of ±2 in the last 3 dims
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in [-2, 2]:
                for sj in [-2, 2]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    arr = np.array(points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr
