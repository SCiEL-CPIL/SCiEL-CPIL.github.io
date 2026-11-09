# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a set of eleven-dimensional integer points satisfying:
      - shape (k, 11)
      - k >= 2
      - no zero vector
      - min pairwise squared distance >= max squared norm
    Uses greedy search over vectors in {-1,0,1}^11 with various max norm^2.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin before 630s kill

    d = 11
    best_points = None
    best_k = 0

    # Generate all vectors in {-1, 0, 1}^d with weight 1 to 4
    all_vectors = []
    norms_sq_list = []
    for w in range(1, 5):
        for positions in combinations(range(d), w):
            for signs in product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                all_vectors.append(v)
                norms_sq_list.append(w)

    all_vectors = np.array(all_vectors, dtype=np.int64)
    norms_sq = np.array(norms_sq_list, dtype=np.int64)

    # Try different max norm^2 values
    for N in [2, 4, 3, 6]:
        if time.time() > deadline - 60.0:
            break

        mask = norms_sq <= N
        candidates = all_vectors[mask]
        cand_norms = norms_sq[mask]

        # Sort by norm^2 ascending
        order = np.argsort(cand_norms)
        candidates = candidates[order]

        selected = []
        for i in range(len(candidates)):
            if time.time() > deadline - 10.0:
                break
            v = candidates[i]
            if not selected:
                selected.append(v)
                continue
            # Check distance to all selected
            diffs = np.array(selected) - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            if np.min(dists_sq) >= N:
                selected.append(v)

        k = len(selected)
        if k > best_k:
            best_k = k
            best_points = np.array(selected, dtype=np.int64)

    # Fallback to the 220-point solution if nothing better found
    if best_points is None or best_k < 2:
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        best_points = np.array(points_list, dtype=np.int64)

    best_points = np.around(best_points).astype(np.int64)
    return best_points


# EVOLVE-BLOCK-END
