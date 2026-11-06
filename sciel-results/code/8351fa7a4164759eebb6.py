"""No-isosceles set: conflict-graph branch-and-bound with aggressive pruning."""
import numpy as np
import random
import time

N = 64
TIME_LIMIT = 280.0
TOTAL = N * N


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0

    # Grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    xs = all_points[:, 0].astype(np.int32)
    ys = all_points[:, 1].astype(np.int32)

    # Pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = (dx * dx + dy * dy).astype(np.int32)
    np.fill_diagonal(dist, 0)

    def check_valid(idx, selected_set):
        """Check if idx can be added to selected_set (a set of indices)."""
        if len(selected_set) < 2:
            return True
        sel = list(selected_set)
        d_to_sel = dist[idx, sel]
        # Check 1: all distances from idx to selected must be distinct
        if len(set(d_to_sel.tolist())) != len(d_to_sel):
            return False
        # Check 2: for each pair (i,j) in selected, ensure d(idx,i) != d(i,j) and d(idx,j) != d(i,j)
        for ii in range(len(sel)):
            for jj in range(ii + 1, len(sel)):
                dij = dist[sel[ii], sel[jj]]
                if d_to_sel[ii] == dij or d_to_sel[jj] == dij:
                    return False
        return True

    def greedy_build(order, deadline):
        """Greedy build following order."""
        selected = set()
        for idx in order:
            if time.time() > deadline:
                break
            if check_valid(idx, selected):
                selected.add(idx)
        return selected

    best_selected = set()
    best_size = 0

    # Multiple greedy passes with different orderings
    num_passes = 30
    for p in range(num_passes):
        if time.time() > deadline:
            break

        if p % 6 == 0:
            order = np.arange(TOTAL)
            np.random.shuffle(order)
        elif p % 6 == 1:
            # Sort by x then y with randomization
            order = np.argsort(xs * N + ys + np.random.rand(TOTAL) * 0.3)
        elif p % 6 == 2:
            # Sort by y then x with randomization
            order = np.argsort(ys * N + xs + np.random.rand(TOTAL) * 0.3)
        elif p % 6 == 3:
            # Sort by distance from center
            cx, cy = N // 2, N // 2
            d_from_center = (xs - cx) ** 2 + (ys - cy) ** 2
            order = np.argsort(d_from_center + np.random.rand(TOTAL) * 0.5)
        elif p % 6 == 4:
            # Sort by distance from corners
            d_corners = np.minimum.reduce([
                (xs - 0) ** 2 + (ys - 0) ** 2,
                (xs - (N-1)) ** 2 + (ys - 0) ** 2,
                (xs - 0) ** 2 + (ys - (N-1)) ** 2,
                (xs - (N-1)) ** 2 + (ys - (N-1)) ** 2
            ])
            order = np.argsort(d_corners + np.random.rand(TOTAL) * 0.5)
        else:
            # Spiral-like ordering from center
            order = np.arange(TOTAL)
            np.random.shuffle(order)

        selected = greedy_build(order, deadline)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = set(selected)

    # Local search: try adding points not in best set
    if time.time() < deadline:
        candidates = list(set(range(TOTAL)) - best_selected)
        random.shuffle(candidates)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in candidates:
                if time.time() > deadline:
                    break
                if check_valid(idx, best_selected):
                    best_selected.add(idx)
                    best_size += 1
                    improved = True
                    candidates.remove(idx)
                    break  # Restart with updated set

        # Swap-based local search
        if time.time() < deadline:
            swap_improved = True
            while swap_improved and time.time() < deadline:
                swap_improved = False
                selected_list = list(best_selected)
                unselected_list = list(set(range(TOTAL)) - best_selected)
                if len(unselected_list) == 0 or len(selected_list) < 2:
                    break

                for _ in range(200):
                    if time.time() > deadline:
                        break
                    if len(unselected_list) == 0:
                        break
                    s_idx = random.choice(selected_list)
                    u_idx = random.choice(unselected_list)

                    # Temporarily remove s_idx
                    best_selected.discard(s_idx)
                    if check_valid(u_idx, best_selected):
                        best_selected.add(u_idx)
                        selected_list = list(best_selected)
                        unselected_list = list(set(range(TOTAL)) - best_selected)
                        swap_improved = True
                    else:
                        best_selected.add(s_idx)

    # Convert to list of tuples
    selected_indices = list(best_selected)
    result = [(int(all_points[i, 0]), int(all_points[i, 1])) for i in selected_indices]
    return result
