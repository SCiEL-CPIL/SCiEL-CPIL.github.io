import numpy as np
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []

    # Weight-2 vectors: exactly two coordinates are ±1, rest 0
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)

    # Weight-3 vectors: exactly three coordinates are ±1, rest 0
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1, s2, s3 in product((1, -1), repeat=3):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    points.append(v)

    return np.array(points)
