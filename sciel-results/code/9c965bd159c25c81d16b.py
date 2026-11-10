# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq = (grid ** 2).sum(axis=1).astype(np.float64)
    deadline = time.time() + 570

    def can_add(arr, sq_arr, p, sq_p):
        m = len(arr)
        if m < 4:
            return True
        diff = arr - p
        sqd = sq_arr - sq_p
        if m == 4:
            M = np.column_stack([diff, sqd])
            return abs(np.linalg.det(M)) >= 0.5
        idx = list(itertools.combinations(range(m), 4))
        M = np.empty((len(idx), 4, 4), dtype=np.float64)
        for i, c in enumerate(idx):
            M[i, :, :3] = diff[c]
            M[i, :, 3] = sqd[c]
        return not np.any(np.abs(np.linalg.det(M)) < 0.5)

    def greedy(order):
        arr = np.empty((0, 3), dtype=np.int64)
        sq_arr = np.empty(0, dtype=np.float64)
        for p in order:
            pi = int(p[0] * 64 + p[1] * 8 + p[2])
            sq_p = sq[pi]
            if can_add(arr, sq_arr, p, sq_p):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                sq_arr = np.append(sq_arr, sq_p)
        return arr

    np.random.seed(42)
    best = None
    best_len = 0

    for r in range(200):
        if time.time() > deadline:
            break
        order = grid[np.random.permutation(N)]
        arr = greedy(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    if best is None:
        best = grid[:50]

    return best.astype(np.int64)

# EVOLVE-BLOCK-END
