"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=100.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def voronoi_inradii(x, y):
        """
        Compute inradii using Voronoi cells.
        For each point, the inradius is the minimum distance to:
        1. The boundary of the unit square
        2. The Voronoi edges (perpendicular bisectors with neighboring points)
        
        This is equivalent to: min(dist_to_boundary, min_j dist(i,j)/2)
        which is what compute_radii does, but we use this name for clarity.
        """
        return compute_radii(x, y)

    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))  # 6
    def grid_init():
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization, tuned for n=32."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        # Use a 6x6 hex grid to get exactly 36 points, take first 32
        # Spacing chosen so circles fit nicely
        s = 1.0 / 5.5  # ~0.182
        for row in range(6):
            for col in range(6):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                # Center the hex grid
                x0[idx] = x0[idx] - 0.5 + 0.5
                y0[idx] = y0[idx] - 0.5 + 0.5
                x0[idx] = np.clip(x0[idx], 0.02, 0.98)
                y0[idx] = np.clip(y0[idx], 0.02, 0.98)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Powell from multiple initializations
    initializations = [grid_init(), hex_init()]
    for init_params in initializations:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(
                objective, init_params, method='Powell', bounds=bounds,
                options={'maxiter': 6000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 2: Smooth objective optimization with L-BFGS-B from multiple starts
    np.random.seed(42)
    
    # Use smooth objective with L-BFGS-B for gradient-based optimization
    for restart in range(12):
        if time.time() > deadline - 40:
            break

        if restart == 0:
            params_start = best_params.copy()
        elif restart == 1:
            params_start = grid_init()
        elif restart == 2:
            params_start = hex_init()
        elif restart == 3:
            # Rotated hex
            x0, y0 = hex_init()[0::2], hex_init()[1::2]
            # Rotate 30 degrees
            cx, cy = 0.5, 0.5
            angle = np.pi / 6
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
            yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        else:
            # Jittered grid
            x_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.05) / side
            y_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.05) / side
            params_start = np.zeros(2 * N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    params_start[2 * idx] = np.clip(x_rand[i], 0.01, 0.99)
                    params_start[2 * idx + 1] = np.clip(y_rand[j], 0.01, 0.99)
                    idx += 1
            if idx < N:
                for k in range(idx, N):
                    params_start[2 * k] = 0.5
                    params_start[2 * k + 1] = 0.5

        try:
            # Use smooth objective with L-BFGS-B
            res = minimize(
                smooth_objective, params_start, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            # Evaluate with the true (non-smooth) objective
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_params = res.x
                best_obj = true_obj
        except Exception:
            pass

    # Phase 3: Powell refinement from best smooth result
    if time.time() < deadline - 30:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 6000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Local perturbation refinement (vectorized for speed)
    if time.time() < deadline - 15:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        radii_best = compute_radii(x_best, y_best)
        sum_best = np.sum(radii_best)

        for _ in range(50):
            if time.time() > deadline - 8:
                break
            improved = False
            for i in range(N):
                for step_size in [0.003, 0.0015, 0.0008, 0.0004]:
                    for dx_try in [-step_size, 0, step_size]:
                        for dy_try in [-step_size, 0, step_size]:
                            if dx_try == 0 and dy_try == 0:
                                continue
                            x_try = x_best.copy()
                            y_try = y_best.copy()
                            x_try[i] += dx_try
                            y_try[i] += dy_try

                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue

                            radii_try = compute_radii(x_try, y_try)
                            sum_try = np.sum(radii_try)

                            if sum_try > sum_best + 1e-15:
                                x_best = x_try
                                y_best = y_try
                                radii_best = radii_try
                                sum_best = sum_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break

        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = objective(best_params)

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 10000, 'xtol': 1e-15, 'ftol': 1e-15}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
