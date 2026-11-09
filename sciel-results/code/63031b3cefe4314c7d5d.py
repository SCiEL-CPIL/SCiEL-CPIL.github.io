# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    Select as many points as possible from {0,...,7}^3 such that no five
    selected points are co-spherical (or co-planar).
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 20.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)

    # Precompute a lookup: for each grid point index, its squared norm
    # We'll work with indices into grid for speed, then convert to coordinates at the end

    def can_add(arr_idx: np.ndarray, p_idx: int) -> bool:
        """Check if point p_idx can be added to set arr_idx without creating 5 co-spherical."""
        m = len(arr_idx)
        if m < 4:
            return True
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        arr = grid[arr_idx]
        arr_sq = grid_sq[arr_idx]

        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        # For small m, sequential with early termination
        # For larger m, batch if C(m,4) is not too large
        from math import comb
        c4 = comb(m, 4)
        if c4 <= 5000:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            sub = w[combos]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Sequential with early termination
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    def greedy_pass(order: np.ndarray, deadline: float) -> tuple:
        """Run a greedy pass over the given order of point indices."""
        arr_idx = []
        skipped = []

        for p_idx in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            if can_add(np.array(arr_idx, dtype=np.int64) if arr_idx else np.empty(0, dtype=np.int64), p_idx):
                arr_idx.append(p_idx)
            else:
                skipped.append(p_idx)

        return np.array(arr_idx, dtype=np.int64), skipped

    def local_improve(arr_idx: np.ndarray, skipped: list, deadline: float) -> np.ndarray:
        """Try to insert previously skipped points into arr_idx."""
        arr_idx = arr_idx.copy()
        improved = True
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            new_skipped = []
            for p_idx in skipped:
                if time.monotonic() + safety_margin >= deadline:
                    new_skipped.append(p_idx)
                    continue
                if can_add(arr_idx, p_idx):
                    arr_idx = np.append(arr_idx, p_idx)
                    improved = True
                else:
                    new_skipped.append(p_idx)
            skipped = new_skipped
        return arr_idx

    rng = np.random.default_rng(0)
    best_idx = None
    num_restarts = 100  # upper bound; actual count limited by deadline

    for restart in range(num_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = rng.permutation(N)
        arr_idx, skipped = greedy_pass(order, deadline)
        arr_idx = local_improve(arr_idx, skipped, deadline)

        if best_idx is None or len(arr_idx) > len(best_idx):
            best_idx = arr_idx

    if best_idx is None:
        best_idx = np.empty(0, dtype=np.int64)

    return grid[best_idx].astype(np.int64)


# EVOLVE-BLOCK-END
