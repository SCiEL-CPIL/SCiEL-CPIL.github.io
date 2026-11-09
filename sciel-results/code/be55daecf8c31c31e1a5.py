# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    # Precompute all C(N,4) combos? No, too many. But we can precompute index lists for can_add.
    # We'll compute combos on the fly but vectorize the det.

    def can_add(arr_indices, p_idx):
        """
        arr_indices: list/array of indices of points already in the set.
        p_idx: index of candidate point to add.
        Returns True if adding p_idx keeps the set valid.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        p = grid[p_idx]
        p_sq = grid_sq[p_idx]

        # Build matrix: rows[i] = (arr[i] - p, |arr[i]|^2 - |p|^2)
        # arr_indices is a list of grid indices
        arr_pts = grid[arr_indices]  # (m, 3)
        diffs = arr_pts.astype(np.float64) - p.astype(np.float64)  # (m, 3)
        sq_diffs = grid_sq[arr_indices] - p_sq  # (m,)
        rows = np.column_stack([diffs, sq_diffs])  # (m, 4)

        # Check all C(m,4) combinations
        if m <= 25:
            # Batch: get all 4-combos, index into rows, batch det
            idx = np.arange(m)
            combos = list(itertools.combinations(idx, 4))
            if not combos:
                return True
            combo_arr = np.array(combos)  # (C, 4)
            sub = rows[combo_arr]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # For larger m, check incrementally with early termination
            for combo in itertools.combinations(range(m), 4):
                sub = rows[list(combo)]
                det_val = np.linalg.det(sub)
                if abs(det_val) < 0.5:
                    return False
            return True

    def greedy_fill(order):
        """Greedy fill: add points in given order if valid."""
        arr_indices = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            if can_add(arr_indices, idx):
                arr_indices.append(idx)
        return arr_indices

    def local_search(arr_indices, max_rounds=50):
        """
        Swap-based local search: try to improve by removing one point and adding another.
        Returns improved list.
        """
        if len(arr_indices) < 4:
            return arr_indices

        current = list(arr_indices)
        current_set = set(current)
        improved = True
        rounds = 0

        while improved and rounds < max_rounds and time.time() < deadline - 30:
            improved = False
            rounds += 1
            for i in range(len(current)):
                if time.time() > deadline - 30:
                    break
                # Remove point i
                remaining = [x for x in current if x != current[i]]
                remaining_set = set(remaining)

                # Try adding each grid point not in remaining_set
                for j in range(N):
                    if j in remaining_set:
                        continue
                    if can_add(remaining, j):
                        new_list = remaining + [j]
                        new_set = set(new_list)
                        if new_set != current_set:
                            current = new_list
                            current_set = new_set
                            improved = True
                            break  # restart scan after a swap
                # Continue scanning other removals in this round
        return current

    # --- Main search loop ---
    rng = np.random.default_rng(42)
    best_set = []
    best_size = 0

    num_restarts = 200

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Vary ordering strategy
        r = restart % 4
        if r == 0:
            order = rng.permutation(N)
        elif r == 1:
            # Bias toward corners (higher dist from center)
            dist = np.sum((grid.astype(np.float64) - 3.5) ** 2, axis=1)
            weights = dist + 1e-9
            weights /= weights.sum()
            order = rng.choice(N, size=N, replace=False, p=weights)
        elif r == 2:
            # Bias toward center
            dist = np.sum((grid.astype(np.float64) - 3.5) ** 2, axis=1)
            weights = (dist.max() + 1.0 - dist) + 1e-9
            weights /= weights.sum()
            order = rng.choice(N, size=N, replace=False, p=weights)
        else:
            # Sort by some hash for deterministic variety
            order = np.argsort(rng.random(N))

        # Greedy fill
        current = greedy_fill(order)

        # Local search improvement
        if len(current) >= 4:
            current = local_search(current, max_rounds=30)

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

        # Simulated annealing phase on current best
        if len(current) >= 4 and time.time() < deadline - 60:
            sa_set = list(current)
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            sa_steps = 0
            sa_max_steps = 5000

            while sa_steps < sa_max_steps and time.time() < deadline - 30:
                sa_steps += 1
                move_type = rng.random()

                if move_type < 0.4:
                    # Try add
                    candidates = [i for i in range(N) if i not in sa_set]
                    if candidates:
                        move_idx = int(rng.choice(candidates))
                        if can_add(sa_set, move_idx):
                            sa_set.append(move_idx)
                            if len(sa_set) > best_size:
                                best_size = len(sa_set)
                                best_set = list(sa_set)
                elif move_type < 0.7:
                    # Try remove
                    if sa_set:
                        move_idx = int(rng.choice(len(sa_set)))
                        if rng.random() < np.exp(-1.0 / T):
                            sa_set.pop(move_idx)
                else:
                    # Try swap
                    if sa_set and len(sa_set) < N:
                        out_pos = int(rng.choice(len(sa_set)))
                        out_idx = sa_set[out_pos]
                        candidates = [i for i in range(N) if i not in sa_set]
                        if candidates:
                            in_idx = int(rng.choice(candidates))
                            # Check: remove out_idx, add in_idx
                            remaining = [x for x in sa_set if x != out_idx]
                            if can_add(remaining, in_idx):
                                sa_set[out_pos] = in_idx

                T *= cooling
                if T < min_T:
                    T = min_T

    # Convert best set to array
    if best_size > 0:
        result = np.array([grid[i] for i in best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
