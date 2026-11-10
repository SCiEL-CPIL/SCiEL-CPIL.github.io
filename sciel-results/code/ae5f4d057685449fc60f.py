import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five selected points are co-spherical (or co-planar).
    Returns a plain NumPy array of shape (k, 3) with integer dtype.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    r2 = (grid ** 2).sum(axis=1)

    # Precompute combinations for m up to a reasonable limit
    max_m = 25
    combos_cache = {}
    for m in range(4, max_m + 1):
        combos_cache[m] = list(itertools.combinations(range(m), 4))

    start_time = time.time()
    deadline = start_time + 590

    best_set = []
    best_len = 0

    def can_add(p_idx, S_indices):
        """
        Check if p_idx can be added to the set defined by S_indices.
        S_indices is a list of indices into the grid.
        Uses early-exit to avoid computing all combinations when possible.
        """
        m = len(S_indices)
        if m < 4:
            return True

        p = grid[p_idx]
        rp2 = r2[p_idx]

        # Get precomputed combinations for this size m
        if m in combos_cache:
            combos = combos_cache[m]
        else:
            combos = list(itertools.combinations(range(m), 4))

        num_combos = len(combos)
        if num_combos == 0:
            return True

        # Process combinations in chunks for efficiency, with early exit
        # For small m, just do it all at once
        if num_combos <= 100:
            mats = np.empty((num_combos, 4, 4), dtype=np.float64)
            for k, combo in enumerate(combos):
                s_idxs = [S_indices[i] for i in combo]
                qs = grid[s_idxs]
                qr2s = r2[s_idxs]
                mats[k, :, :3] = qs - p
                mats[k, :, 3] = qr2s - rp2
            dets = np.linalg.det(mats)
            if np.any(np.abs(dets) < 0.5):
                return False
            return True
        else:
            # Process in chunks of 64
            chunk_size = 64
            for start in range(0, num_combos, chunk_size):
                end = min(start + chunk_size, num_combos)
                chunk = combos[start:end]
                mats = np.empty((len(chunk), 4, 4), dtype=np.float64)
                for k, combo in enumerate(chunk):
                    s_idxs = [S_indices[i] for i in combo]
                    qs = grid[s_idxs]
                    qr2s = r2[s_idxs]
                    mats[k, :, :3] = qs - p
                    mats[k, :, 3] = qr2s - rp2
                dets = np.linalg.det(mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    np.random.seed(42)

    # Multiple random restarts with different strategies
    for restart in range(200):
        if time.time() >= deadline:
            break

        # Alternate between random permutation and some heuristic ordering
        if restart % 3 == 0:
            order = np.random.permutation(N)
        elif restart % 3 == 1:
            # Try to order by some property (e.g., distance from center)
            center = 3.5
            dists = np.sqrt(((grid - center) ** 2).sum(axis=1))
            order = np.argsort(dists + np.random.randn(N) * 0.5)
        else:
            # Random but biased toward corners/edges
            weights = 1.0 / (1.0 + 0.1 * (grid.sum(axis=1) + (7 - grid).sum(axis=1)))
            order = np.random.choice(N, size=N, replace=False, p=weights / weights.sum())

        S_indices = []

        for p_idx in order:
            # Early stop if we've already beaten best and are doing well
            if can_add(p_idx, S_indices):
                S_indices.append(p_idx)

        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()

            # If we find a good set, we can try to extend it greedily
            # but first save this as our best

    # Final attempt: try to extend the best set found
    if best_set and best_len < N:
        # Create a set of indices for quick lookup
        best_set_set = set(best_set)
        # Try adding remaining points
        for p_idx in range(N):
            if p_idx in best_set_set:
                continue
            if time.time() >= deadline - 5:
                break
            if can_add(p_idx, best_set):
                best_set.append(p_idx)
                best_set_set.add(p_idx)
                best_len = len(best_set)

    if not best_set:
        return grid[:4].astype(np.int64)

    return grid[best_set].astype(np.int64)
