import numpy as np
import time
import heapq

class Optimizer:
    """Adaptive lattice refinement optimizer for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # ---- Phase 1: Initial grid evaluation ----
        # Evaluate a coarse grid to identify promising regions
        # For dim=5, use a 2^5 = 32 point grid initially
        grid_size = 2
        grid_points = []
        for i in range(grid_size ** dim):
            coords = []
            n = i
            for d in range(dim):
                coords.append(n % grid_size)
                n //= grid_size
            # Map to [-5, 5]
            x = np.array([(2 * c / (grid_size - 1) - 1) * 5 for c in coords])
            grid_points.append(x)

        best_x = None
        best_f = np.inf
        evaluated = {}  # point -> function value

        for x in grid_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            evaluated[tuple(np.round(x, 6))] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: Adaptive subdivision ----
        # Use a priority queue to refine the most promising regions
        # Each cell is defined by its center and half-widths
        # Priority is based on the function value at the center

        # Initialize cells from the grid
        cells = []
        cell_id = 0
        for x in grid_points:
            # Half-width is 5/grid_size
            hw = 5.0 / grid_size
            half_widths = np.full(dim, hw)
            heapq.heappush(cells, (best_f, cell_id, x.copy(), half_widths))  # min-heap
            cell_id += 1

        # Track evaluated points to avoid duplicates
        eval_set = set(evaluated.keys())

        # Adaptive refinement loop
        max_depth = 8  # limit subdivision depth
        min_half_width = 5.0 / (grid_size * (2 ** max_depth))

        while calls[0] < budget and time.time() < deadline:
            if not cells:
                break

            # Pop the cell with the best (lowest) function value
            f_val, cid, center, half_widths = heapq.heappop(cells)

            # Check if we should subdivide this cell
            if np.min(half_widths) < min_half_width:
                # Too small, skip
                continue

            # Check time and budget
            if time.time() > deadline or calls[0] >= budget:
                break

            # Determine which dimension to split (largest extent)
            split_dim = int(np.argmax(half_widths))

            # Create two child cells by splitting along split_dim
            new_hw1 = half_widths.copy()
            new_hw2 = half_widths.copy()
            new_hw1[split_dim] /= 2
            new_hw2[split_dim] /= 2

            center1 = center.copy()
            center2 = center.copy()
            center1[split_dim] -= half_widths[split_dim] / 2
            center2[split_dim] += half_widths[split_dim] / 2

            # Clip to bounds
            center1 = np.clip(center1, -5, 5)
            center2 = np.clip(center2, -5, 5)

            # Evaluate both children
            for child_center in [center1, center2]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                key = tuple(np.round(child_center, 6))
                if key in eval_set:
                    continue
                try:
                    fx = safe_func(child_center)
                except StopIteration:
                    break
                eval_set.add(key)
                if fx < best_f:
                    best_f = fx
                    best_x = child_center.copy()
                # Push child cell to priority queue
                heapq.heappush(cells, (fx, cell_id, child_center.copy(), new_hw1 if child_center is center1 else new_hw2))
                cell_id += 1

        # ---- Phase 3: Local polish around best point ----
        polish_calls_remaining = budget - calls[0]
        if polish_calls_remaining > 5 and time.time() < deadline:
            center = best_x.copy()
            cf = best_f
            sigma = 0.1

            # Coordinate-wise refinement
            for _ in range(min(30, polish_calls_remaining // 3)):
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline:
                        break
                    for sign in (1, -1):
                        if calls[0] >= budget:
                            break
                        x2 = center.copy()
                        x2[d] = np.clip(center[d] + sign * sigma, -5, 5)
                        key = tuple(np.round(x2, 6))
                        if key in eval_set:
                            continue
                        try:
                            f2 = safe_func(x2)
                        except StopIteration:
                            break
                        eval_set.add(key)
                        if f2 < cf:
                            cf = f2
                            center = x2
                            improved = True
                    if time.time() > deadline:
                        break
                if improved:
                    sigma *= 0.7
                else:
                    sigma *= 0.5
                if sigma < 1e-8:
                    break

            best_x = center
            best_f = cf

        return best_x, best_f
