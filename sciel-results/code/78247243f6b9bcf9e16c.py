import numpy as np
import itertools

def construct():
    D = 11
    vecs = []
    # Generate all vectors of weight 2 (two non-zero entries, each +/- 1)
    # Norm is sqrt(2).
    # Min distance between distinct weight-2 vectors is sqrt(2) or 2.
    # sqrt(2) >= sqrt(2) holds.
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vecs.append(v)
    return np.array(vecs, dtype=int)
