"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Objective: maximize sum of radii
    # Constraints:
    #   1. x_i - r_i >= 0, x_i + r_i <= 1
    #   2. y_i - r_i >= 0, y_i + r_i <= 1
    #   3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j

    # Strategy: Use a simulated annealing / gradient-based approach
    # Represent state as (x, y, r) for each circle
    # Use scipy.optimize or manual gradient descent with penalty method

    from scipy.optimize import minimize, constraints as sp_constraints

    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    xs = np.array([(i + 0.5) / side for i in range(side) for j in range(side)])[:N]
    ys = np.array([(j + 0.5) / side for i in range(side) for j in range(side)])[:N]
    rs = np.full(N, 0.5 / side - 1e-3)
    x0 = np.concatenate([xs, ys, rs])

    # Penalty-based optimization
    def objective(v):
        """Negative sum of radii plus penalties for constraint violations."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]

        # Sum of radii (we maximize, so minimize negative)
        obj = -np.sum(r)

        # Boundary penalties
        # x - r >= 0 => r - x <= 0
        # x + r <= 1 => x + r - 1 <= 0
        # Same for y
        pen_x_left = np.maximum(0, r - x) ** 2
        pen_x_right = np.maximum(0, x + r - 1) ** 2
        pen_y_left = np.maximum(0, r - y) ** 2
        pen_y_right = np.maximum(0, y + r - 1) ** 2
        boundary_pen = np.sum(pen_x_left) + np.sum(pen_x_right) + np.sum(pen_y_left) + np.sum(pen_y_right)

        # Circle-circle overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    overlap_pen += (min_dist - np.sqrt(dist_sq)) ** 2

        return obj + 1000.0 * (boundary_pen + overlap_pen)

    # Run multiple restarts with different initial configurations
    best_result = None
    best_obj = np.inf

    # Try several different initial layouts
    initial_configs = []

    # Config 1: Uniform grid
    initial_configs.append(x0.copy())

    # Config 2: Hexagonal-like packing
    if N <= 25:
        # 5x5 grid with hexagonal offset
        hex_xs = []
        hex_ys = []
        for row in range(5):
            for col in range(5):
                if len(hex_xs) < N:
                    x_pos = (col + 0.5) / 5 + (0.25 if row % 2 == 1 else 0) / 5
                    y_pos = (row + 0.5) / 5
                    hex_xs.append(x_pos)
                    hex_ys.append(y_pos)
        if len(hex_xs) == N:
            hex_rs = np.full(N, 0.15)
            initial_configs.append(np.concatenate([np.array(hex_xs), np.array(hex_ys), hex_rs]))

    # Config 3: Random initialization
    for trial in range(5):
        rng = np.random.default_rng(seed=trial + 42)
        rx = rng.uniform(0.15, 0.85, N)
        ry = rng.uniform(0.15, 0.85, N)
        rr = rng.uniform(0.05, 0.18, N)
        initial_configs.append(np.concatenate([rx, ry, rr]))

    for config_idx, x_init in enumerate(initial_configs):
        if time.time() > deadline:
            break

        try:
            result = minimize(objective, x_init, method='L-BFGS-B',
                             options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            continue

    # If we found a good result, use it; otherwise fall back to grid
    if best_result is not None:
        x = best_result[:N]
        y = best_result[N:2*N]
        r = best_result[2*N:3*N]

        # Ensure validity: clip to boundaries and fix overlaps
        # Simple fix: shrink radii slightly and adjust positions
        r = np.maximum(r, 1e-6)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)

        # Check for remaining overlaps and shrink if needed
        for _ in range(10):
            max_violation = 0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        max_violation = max(max_violation, min_dist - dist)

            if max_violation < 1e-6:
                break

            # Shrink all radii slightly
            r *= (1 - 0.01)
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)

        return np.column_stack([x, y, r])
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
