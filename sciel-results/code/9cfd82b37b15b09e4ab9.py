"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin (150s limit)

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

    def smooth_objective(params, alpha=150.0):
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

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
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
        s = 1.0 / 5.5
        for row in range(6):
            for col in range(6):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.02, 0.98)
                y0[idx] = np.clip(y0[idx], 0.02, 0.98)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init_scaled():
        """Hexagonal lattice scaled to fit in unit square, taking N closest to center."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        params = np.zeros(2 * N)
        params[0::2] = init_pts[:, 0]
        params[1::2] = init_pts[:, 1]
        return params

    def random_init(seed):
        rng = np.random.RandomState(seed)
        params = np.zeros(2 * N)
        params[0::2] = rng.uniform(0.05, 0.95, N)
        params[1::2] = rng.uniform(0.05, 0.95, N)
        return params

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Powell from multiple initializations
    initializations = [grid_init(), hex_init(), hex_init_scaled()]
    for init_params in initializations:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                objective, init_params, method='Powell', bounds=bounds,
                options={'maxiter': 4000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 2: Smooth objective optimization with L-BFGS-B from multiple starts
    np.random.seed(42)
    
    for restart in range(14):
        if time.time() > deadline - 50:
            break

        if restart == 0:
            params_start = best_params.copy()
        elif restart == 1:
            params_start = grid_init()
        elif restart == 2:
            params_start = hex_init()
        elif restart == 3:
            params_start = hex_init_scaled()
        elif restart == 4:
            # Rotated hex
            x0, y0 = hex_init_scaled()[0::2], hex_init_scaled()[1::2]
            cx, cy = 0.5, 0.5
            angle = np.pi / 6
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
            yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        elif restart == 5:
            params_start = random_init(100)
        elif restart == 6:
            params_start = random_init(200)
        elif restart == 7:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
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
        else:
            # Jittered best
            params_start = best_params + np.random.normal(0, 0.02, 2 * N)
            params_start = np.clip(params_start, 0.01, 0.99)

        try:
            res = minimize(
                smooth_objective, params_start, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_params = res.x
                best_obj = true_obj
        except Exception:
            pass

    # Phase 3: Powell refinement from best smooth result
    if time.time() < deadline - 35:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 5000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Local refinement using Nelder-Mead on true objective
    if time.time() < deadline - 25:
        for nm_restart in range(6):
            if time.time() > deadline - 12:
                break
            try:
                if nm_restart == 0:
                    p_start = best_params.copy()
                elif nm_restart == 1:
                    p_start = best_params + np.random.normal(0, 0.001, 2 * N)
                    p_start = np.clip(p_start, 0.001, 0.999)
                elif nm_restart == 2:
                    p_start = best_params + np.random.normal(0, 0.005, 2 * N)
                    p_start = np.clip(p_start, 0.001, 0.999)
                else:
                    p_start = best_params + np.random.normal(0, 0.002, 2 * N)
                    p_start = np.clip(p_start, 0.001, 0.999)
                
                res = minimize(
                    objective, p_start, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_obj:
                    best_params = res.x
                    best_obj = res.fun
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 8:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 12000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

        if time.time() < deadline - 3:
            try:
                res = minimize(
                    objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
