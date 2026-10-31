"""Program for kissing number in dimension 11: maximize k nonzero integer vectors
with min pairwise distance >= max norm.

Strategy: Use support-based greedy with weight-3 vectors. For weight w=3, norm^2=3,
constraint is u·v <= 1 for all pairs. Conflicts only arise when supports intersect
in 2 or 3 indices with matching signs. Greedily select sign patterns per support.
"""
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0

    D = 11
    w = 3

    supports = list(itertools.combinations(range(D), w))
    all_patterns = list(itertools.product([1, -1], repeat=w))

    selected_vecs = []
    selected_vecs_arr = np.empty((0, D), dtype=np.int64)

    for support in supports:
        if time.time() > deadline:
            break

        for pattern in all_patterns:
            if time.time() > deadline:
                break

            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, pattern):
                v[idx] = s

            if selected_vecs_arr.shape[0] > 0:
                dots = selected_vecs_arr @ v
                if np.any(dots > 1):
                    continue

            selected_vecs.append(v)
            selected_vecs_arr = np.vstack([selected_vecs_arr, v.reshape(1, -1)])

    if len(selected_vecs) == 0:
        # Fallback: D11
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        vec = [0] * D
                        vec[i] = si
                        vec[j] = sj
                        vectors.append(vec)
        return np.array(vectors, dtype=np.int64)

    return np.array(selected_vecs, dtype=np.int64)
