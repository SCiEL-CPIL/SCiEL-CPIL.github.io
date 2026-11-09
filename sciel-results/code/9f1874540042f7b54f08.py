import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    deadline = time.time() + 590
    rng = np.random.default_rng(42)

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True

        p_sq = float((p ** 2).sum())
        arr_sq = (arr ** 2).sum(axis=1).astype(np.float64)
        diff = (arr - p).astype(np.float64)
        sq_diff = arr_sq - p_sq

        for i0 in range(m - 3):
            d0 = diff[i0]
            s0 = sq_diff[i0]
            for i1 in range(i0 + 1, m - 2):
                d1 = diff[i1]
                s1 = sq_diff[i1]
                for i2 in range(i1 + 1, m - 1):
                    d2 = diff[i2]
                    s2 = sq_diff[i2]
                    for i3 in range(i2 + 1, m):
                        d3 = diff[i3]
                        s3 = sq_diff[i3]

                        # Determinant of [[d0x,d0y,d0z,s0],
                        #                  [d1x,d1y,d1z,s1],
                        #                  [d2x,d2y,d2z,s2],
                        #                  [d3x,d3y,d3z,s3]]
                        # Using cofactor expansion along last column:
                        # det = s0*det(d1,d2,d3) - s1*det(d0,d2,d3) + s2*det(d0,d1,d3) - s3*det(d0,d1,d2)
                        # where det(d1,d2,d3) = d1 dot (d2 cross d3)

                        c01 = np.cross(d2, d3)
                        c02 = np.cross(d2, d1)
                        c03 = np.cross(d1, d3)
                        c12 = np.cross(d0, d3)
                        c13 = np.cross(d0, d2)

                        det_val = (s0 * np.dot(d1, c01)
                                   - s1 * np.dot(d0, c02)
                                   + s2 * np.dot(d0, c03)
                                   - s3 * (np.dot(d0, np.cross(d1, d2))))

                        if abs(det_val) < 0.5:
                            return False

        return True

    def greedy_build(order: np.ndarray, start_set: np.ndarray = None) -> np.ndarray:
        if start_set is None:
            curr_set = np.empty((0, 3), dtype=np.int64)
        else:
            curr_set = start_set.copy()

        for p in order:
            if time.time() > deadline:
                break
            if is_valid_addition(curr_set, p):
                if len(curr_set) > 0:
                    curr_set = np.vstack([curr_set, p])
                else:
                    curr_set = p[None, :]

        return curr_set

    def local_search(current_set: np.ndarray) -> np.ndarray:
        current_set = current_set.copy()
        current_set_set = set(tuple(p) for p in current_set)
        no_improve_count = 0
        max_no_improve = 30

        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False

            # Try to add any remaining point
            remaining = [p for p in grid if tuple(p) not in current_set_set]
            for p in remaining:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_set, p):
                    current_set = np.vstack([current_set, p])
                    current_set_set.add(tuple(p))
                    improved = True
                    no_improve_count = 0
                    break

            if improved:
                continue

            # Try swaps
            if len(current_set) > 0 and len(remaining) > 0:
                # More systematic swap: try removing each point and adding a random remaining point
                swap_attempts = min(100, len(current_set) * min(len(remaining), 20))
                for _ in range(swap_attempts):
                    if time.time() > deadline:
                        break

                    in_idx = int(rng.integers(len(current_set)))
                    p_in = current_set[in_idx]

                    out_idx = int(rng.integers(len(remaining)))
                    p_out = remaining[out_idx]

                    reduced_set = np.delete(current_set, in_idx, axis=0)

                    if is_valid_addition(reduced_set, p_out):
                        current_set = np.vstack([reduced_set, p_out])
                        current_set_set.discard(tuple(p_in))
                        current_set_set.add(tuple(p_out))
                        remaining = [p for p in grid if tuple(p) not in current_set_set]
                        improved = True
                        no_improve_count = 0
                        break

            if not improved:
                no_improve_count += 1

        return current_set

    best_set = np.empty((0, 3), dtype=np.int64)

    # Phase 1: Greedy with multiple random orderings
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        perm = rng.permutation(num_points_total)
        order = grid[perm]
        curr_set = greedy_build(order)

        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local search from the best greedy result
    if len(best_set) > 0:
        best_set = local_search(best_set)

    # Phase 3: More greedy restarts with the remaining time, each followed by local search
    for restart in range(200):
        if time.time() > deadline:
            break

        perm = rng.permutation(num_points_total)
        order = grid[perm]
        curr_set = greedy_build(order)

        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()
            best_set = local_search(best_set)
        elif len(curr_set) >= len(best_set) - 2 and time.time() < deadline - 30:
            # Even if not strictly better, try local search from this set
            improved_set = local_search(curr_set)
            if len(improved_set) > len(best_set):
                best_set = improved_set.copy()

    return best_set.astype(np.int64)
