"""Improved program for kissing number in dimension 11."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 10s safety margin

    # Generate all type D vectors: (±1, ±1, 0, ..., 0) on each pair of coordinates
    # These have norm sqrt(2), and all pairwise distances are >= sqrt(2)
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)

    arr = np.array(vectors, dtype=np.int64)
    return arr

# EVOLVE-BLOCK-END
