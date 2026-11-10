import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1)

    def check_validity(arr: np.ndarray) -> bool:
        """Check if the set arr is valid (no 5 co-spherical)."""
        m = len(arr)
        if m < 5:
            return True

        # For each point p in arr, check all combinations of 4 other points
        # that come before it in the ordering
        for i in range(4, m):
            p = arr[i]
            # Earlier points
            earlier = arr[:i]
            diff = earlier - p
            sq_diff = sq_norms[:i] - sq_norms[i]

            # Build (i, 4) matrix
            M = np.empty((i, 4), dtype=np.float64)
            M[:, 0:3] = diff
            M[:, 3] = sq_diff

            # Check all 4-combinations of rows
            for combo in itertools.combinations(range(i), 4):
                det_val = np.linalg.det(M[list(combo)])
                if abs(det_val) < 0.5:
                    return False
        return True

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if adding point p to arr keeps the set valid."""
        m = len(arr)
        if m < 4:
            return True

        diff = arr - p
        sq_diff = sq_norms[:m] - sq_norms[np.where((grid == p).all(axis=1))[0][0]]

        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff

        for combo in itertools.combinations(range(m), 4):
            det_val = np.linalg.det(M[list(combo)])
            if abs(det_val) < 0.5:
                return False
        return True

    def greedy_build(order, deadline):
        curr_set = []
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if not curr_set:
                curr_set.append(p)
            else:
                if is_valid_addition(np.array(curr_set), p):
                    curr_set.append(p)
        return np.array(curr_set, dtype=np.int64)

    # Phase 1: Multiple greedy restarts with different orderings
    best_set = np.empty((0, 3), dtype=np.int64)

    np.random.seed(42)

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 60:  # Leave time for local search
            break

        # Different ordering strategies
        if r % 3 == 0:
            order = np.random.permutation(num_points_total)
        elif r % 3 == 1:
            # Sort by some property to get different structures
            # Use a random linear functional
            weights = np.random.randn(3)
            scores = grid @ weights
            order = np.argsort(scores)
        else:
            # Random but biased towards corners
            corner_dist = np.minimum.reduce([
                np.abs(grid - np.array([0,0,0])),
                np.abs(grid - np.array([7,7,7])),
                np.abs(grid - np.array([0,7,0])),
                np.abs(grid - np.array([7,0,7])),
                np.abs(grid - np.array([0,0,7])),
                np.abs(grid - np.array([7,7,0])),
                np.abs(grid - np.array([0,7,7])),
                np.abs(grid - np.array([7,0,0])),
            ]).sum(axis=1)
            # Bias towards corners
            probs = 1.0 / (1.0 + corner_dist)
            probs /= probs.sum()
            order = np.random.choice(num_points_total, size=num_points_total, replace=False, p=probs)

        curr_set = greedy_build(order, deadline - 60)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local search - try to add more points
    if len(best_set) > 0:
        current_set = best_set.copy()
        current_set_list = list(current_set)

        # Build index lookup
        point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
        current_indices = set(point_to_idx[tuple(p)] for p in current_set)
        remaining_indices = list(set(range(num_points_total)) - current_indices)

        # Rejection tracking
        rejection_counts = {}

        no_improve = 0
        max_no_improve = 200

        while time.time() < deadline:
            improved = False

            # Try to add a point
            if remaining_indices:
                # Score candidates by rejection count (lower is better)
                scored = sorted(remaining_indices, key=lambda x: rejection_counts.get(x, 0))
                # Take top 50 and some random
                top_candidates = scored[:50]
                if len(remaining_indices) > 50:
                    extra_pool = set(remaining_indices) - set(top_candidates)
                    extra_list = list(extra_pool)
                    top_candidates.extend(np.random.choice(extra_list, min(20, len(extra_list)), replace=False).tolist())

                np.random.shuffle(top_candidates)

                for idx in top_candidates:
                    p = grid[idx]
                    if is_valid_addition(current_set, p):
                        current_set = np.vstack([current_set, p])
                        current_indices.add(idx)
                        remaining_indices.remove(idx)
                        improved = True
                        no_improve = 0
                        break
                    else:
                        rejection_counts[idx] = rejection_counts.get(idx, 0) + 1

            # Try swaps if no improvement
            if not improved and len(current_indices) > 0 and remaining_indices:
                for _ in range(50):
                    if time.time() > deadline:
                        break

                    # Pick random point to remove
                    in_idx = list(current_indices)[np.random.randint(len(current_indices))]
                    p_in = grid[in_idx]

                    # Pick point to add - prioritize high rejection
                    scored_out = sorted(remaining_indices, key=lambda x: rejection_counts.get(x, 0), reverse=True)
                    top_out = scored_out[:30]
                    if len(remaining_indices) > 30:
                        extra_pool = set(remaining_indices) - set(top_out)
                        extra_list = list(extra_pool)
                        top_out.extend(np.random.choice(extra_list, min(10, len(extra_list)), replace=False).tolist())

                    out_idx = top_out[np.random.randint(len(top_out))]
                    p_out = grid[out_idx]

                    # Check if swap is valid
                    mask = np.array([tuple(row) != tuple(p_in) for row in current_set])
                    reduced_set = current_set[mask]

                    if is_valid_addition(reduced_set, p_out):
                        new_set = np.vstack([reduced_set, p_out])
                        current_set = new_set
                        current_indices.remove(in_idx)
                        current_indices.add(out_idx)
                        remaining_indices.remove(out_idx)
                        remaining_indices.append(in_idx)
                        rejection_counts[out_idx] = 0
                        rejection_counts[in_idx] = rejection_counts.get(in_idx, 0) + 1
                        improved = True
                        no_improve = 0
                        break

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Kick: remove several points
                    if len(current_indices) > 10:
                        kick_size = min(5, len(current_indices) // 5)
                        kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                        for idx in kick_idxs:
                            p = grid[idx]
                            mask = np.array([tuple(row) != tuple(p) for row in current_set])
                            current_set = current_set[mask]
                            current_indices.remove(idx)
                            remaining_indices.append(idx)
                            rejection_counts[idx] = 0
                    no_improve = 0
                    max_no_improve = 300

        best_set = current_set

    return best_set.astype(np.int64)
