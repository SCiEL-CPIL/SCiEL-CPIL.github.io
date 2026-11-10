# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points). Five points p_i = (x_i, y_i, z_i) are co-spherical iff the 5x5
    determinant of rows [x_i, y_i, z_i, x_i^2 + y_i^2 + z_i^2, 1] vanishes.

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Strategy: seeded random-order greedy passes to build an initial set, then
    local search via swaps (remove one selected point, try to add a non-selected
    point) to escape greedy local optima. Deadline guard returns best found
    before the wall-clock limit.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    deadline = time.time() + 600  # safety margin before 630s kill

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # (C, 4, 4)
        return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def greedy_pass(order):
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        return arr

    def local_improve(arr, rng):
        """Try swaps: remove a selected point, try adding a non-selected one."""
        arr = arr.copy()
        selected_set = set(map(tuple, arr.tolist()))
        improved = True
        while improved:
            improved = False
            # Try removing each selected point and adding a random non-selected point
            for i in range(len(arr)):
                if time.time() > deadline:
                    return arr
                # Remove point i
                arr_without = np.delete(arr, i, axis=0)
                removed = arr[i]
                # Try several random candidates
                for _ in range(20):
                    candidate = grid[rng.randint(num_points)]
                    if tuple(candidate) in selected_set:
                        continue
                    if can_add(arr_without, candidate):
                        # Accept the swap
                        arr = np.vstack([arr_without, candidate])
                        selected_set.remove(tuple(removed))
                        selected_set.add(tuple(candidate))
                        improved = True
                        break
        return arr

    np.random.seed(0)
    rng = np.random.RandomState(0)
    best = None
    num_restarts = 10

    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        order = grid[rng.permutation(num_points)]
        arr = greedy_pass(order)
        # Local search phase
        if time.time() < deadline - 10:
            arr = local_improve(arr, rng)
        if best is None or len(arr) > len(best):
            best = arr

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
