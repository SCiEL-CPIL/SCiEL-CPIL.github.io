import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 590  # Safety margin

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Uses vectorized determinant computation for speed.
        """
        m = len(arr)
        if m < 4:
            return True

        # diff[i] = arr[i] - p
        diff = arr - p
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        sq_diff = (arr ** 2).sum(axis=1) - (p ** 2).sum()

        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff

        # For small m, use itertools; for larger m, precompute combos
        if m <= 25:
            for combo in itertools.combinations(range(m), 4):
                sub = M[list(combo)]
                det_val = np.linalg.det(sub)
                if abs(det_val) < 0.5:
                    return False
        else:
            # Precompute all 4-combinations as an array of indices
            # This is expensive to generate each time, so we cache by m
            if not hasattr(is_valid_addition, '_combo_cache'):
                is_valid_addition._combo_cache = {}
            if m not in is_valid_addition._combo_cache:
                is_valid_addition._combo_cache[m] = np.array(
                    list(itertools.combinations(range(m), 4)), dtype=np.int32
                )
            combos = is_valid_addition._combo_cache[m]
            if len(combos) == 0:
                return True
            # Extract all sub-matrices at once: shape (num_combos, 4, 4)
            sub_matrices = M[combos]  # (num_combos, 4, 4)
            # Compute all determinants
            dets = np.linalg.det(sub_matrices)
            if np.any(np.abs(dets) < 0.5):
                return False

        return True

    def get_random_order():
        return np.random.permutation(num_points_total)

    def greedy_build(order):
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    best_set = np.empty((0, 3), dtype=np.int64)

    np.random.seed(42)

    # Phase 1: Multiple Greedy Restarts
    # With vectorized det check, each addition is faster, so we can do more restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = get_random_order()
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search
    current_set = best_set.copy()

    # Build a tuple-to-index map for the grid
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}

    current_indices = set()
    for p in current_set:
        current_indices.add(grid_tuple_to_idx[tuple(p)])

    all_indices = set(range(num_points_total))
    remaining_indices = list(all_indices - current_indices)

    no_improve_count = 0
    max_no_improve = 100

    while time.time() < deadline:
        improved = False

        # 1. Try to Add - check all remaining points
        if remaining_indices:
            shuffled_remaining = remaining_indices.copy()
            np.random.shuffle(shuffled_remaining)
            for idx in shuffled_remaining:
                if time.time() > deadline:
                    break
                p = grid[idx]
                if is_valid_addition(current_set, p):
                    if len(current_set) == 0:
                        current_set = p[None, :]
                    else:
                        current_set = np.vstack([current_set, p])

                    current_indices.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve_count = 0
                    break

        # 2. Try to Swap
        if not improved and len(current_indices) > 0 and remaining_indices:
            swaps_to_try = 100
            tried_swaps = set()
            for _ in range(swaps_to_try):
                if time.time() > deadline:
                    break

                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                out_idx_val = remaining_indices[np.random.randint(len(remaining_indices))]

                swap_key = (in_idx_val, out_idx_val)
                if swap_key in tried_swaps:
                    continue
                tried_swaps.add(swap_key)

                p_in = grid[in_idx_val]
                p_out = grid[out_idx_val]

                in_pos = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in):
                        in_pos = i
                        break

                if in_pos is None:
                    continue

                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]

                if is_valid_addition(reduced_set, p_out):
                    current_set = reduced_set
                    if len(current_set) == 0:
                        current_set = p_out[None, :]
                    else:
                        current_set = np.vstack([current_set, p_out])

                    current_indices.remove(in_idx_val)
                    current_indices.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx_val)

                    improved = True
                    no_improve_count = 0
                    break

        # 3. Try 1-for-2 moves (remove one, add two)
        if not improved and len(current_indices) > 2 and len(remaining_indices) >= 2:
            moves_to_try = 50
            for _ in range(moves_to_try):
                if time.time() > deadline:
                    break

                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                p_in = grid[in_idx_val]

                in_pos = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in):
                        in_pos = i
                        break

                if in_pos is None:
                    continue

                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]

                if len(remaining_indices) < 2:
                    continue

                idx1, idx2 = np.random.choice(remaining_indices, 2, replace=False)
                p1 = grid[idx1]
                p2 = grid[idx2]

                if is_valid_addition(reduced_set, p1):
                    temp_set = np.vstack([reduced_set, p1[None, :]]) if len(reduced_set) > 0 else p1[None, :]
                    if is_valid_addition(temp_set, p2):
                        current_set = np.vstack([temp_set, p2[None, :]]) if len(temp_set) > 0 else p2[None, :]

                        current_indices.remove(in_idx_val)
                        current_indices.add(idx1)
                        current_indices.add(idx2)
                        remaining_indices.remove(idx1)
                        remaining_indices.remove(idx2)
                        remaining_indices.append(in_idx_val)

                        improved = True
                        no_improve_count = 0
                        break

        # 4. Try 2-for-1 moves (remove two, add one) — can help when set is dense
        if not improved and len(current_indices) > 4 and len(remaining_indices) >= 1:
            moves_to_try = 30
            for _ in range(moves_to_try):
                if time.time() > deadline:
                    break

                in_idx1, in_idx2 = np.random.choice(list(current_indices), 2, replace=False)
                p_in1 = grid[in_idx1]
                p_in2 = grid[in_idx2]

                # Find and remove both
                in_pos1 = None
                in_pos2 = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in1):
                        in_pos1 = i
                    elif np.array_equal(current_set[i], p_in2):
                        in_pos2 = i
                if in_pos1 is None or in_pos2 is None:
                    continue

                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos1] = False
                reduced_mask[in_pos2] = False
                reduced_set = current_set[reduced_mask]

                out_idx_val = remaining_indices[np.random.randint(len(remaining_indices))]
                p_out = grid[out_idx_val]

                if is_valid_addition(reduced_set, p_out):
                    current_set = reduced_set
                    if len(current_set) == 0:
                        current_set = p_out[None, :]
                    else:
                        current_set = np.vstack([current_set, p_out])

                    current_indices.remove(in_idx1)
                    current_indices.remove(in_idx2)
                    current_indices.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx1)
                    remaining_indices.append(in_idx2)

                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Kick - remove several random points to escape local optimum
                if len(current_indices) > 10:
                    kick_size = min(6, len(current_indices) // 4)
                    kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                    for idx in kick_idxs:
                        p = grid[idx]
                        mask = np.ones(len(current_set), dtype=bool)
                        for i in range(len(current_set)):
                            if np.array_equal(current_set[i], p):
                                mask[i] = False
                                break
                        current_set = current_set[mask]

                        current_indices.remove(idx)
                        remaining_indices.append(idx)
                    no_improve_count = 0
                    max_no_improve = 200

    return current_set.astype(np.int64)
