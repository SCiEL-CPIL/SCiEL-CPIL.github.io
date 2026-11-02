"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin (150s limit)

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

    def smooth_objective_grad(params, alpha=100.0):
        """Smooth objective with analytical gradient for L-BFGS-B."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        n = len(x)

        # Compute distances
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
        f = -np.sum(radii_soft)

        # Gradient computation
        grad = np.zeros(2 * n)

        # Gradient of radii_soft w.r.t. x[i], y[i]
        # d(radii_soft[i])/dx[i] comes from:
        # 1. Soft-min over boundaries: d(min_boundary_soft[i])/dx[i]
        # 2. Soft-min over distances: d(min_dist_soft[i])/dx[i]
        # 3. Soft-min combining them

        # Boundary gradient contribution
        # d(min_boundary_soft)/dx[i] = -sum_j exp_b[i,j] * d(scaled_b[i,j])/dx[i] / sum_j exp_b[i,j]
        # where scaled_b[i,j] = -alpha * boundary_terms[i,j]
        # d(boundary_terms[i,0])/dx[i] = 1, d(boundary_terms[i,1])/dx[i] = -1, etc.

        # For circle i, boundary terms are [x[i], 1-x[i], y[i], 1-y[i]]
        # d(scaled_b[i,0])/dx[i] = -alpha * 1
        # d(scaled_b[i,1])/dx[i] = -alpha * (-1) = alpha
        # d(scaled_b[i,2])/dy[i] = -alpha * 1
        # d(scaled_b[i,3])/dy[i] = -alpha * (-1) = alpha

        # min_boundary_soft[i] = -log_sum_b[i] / alpha
        # d(min_boundary_soft[i])/dx[i] = -(1/alpha) * d(log_sum_b[i])/dx[i]
        # d(log_sum_b[i])/dx[i] = sum_j exp_b[i,j] * d(scaled_b[i,j])/dx[i] / sum_j exp_b[i,j]

        for i in range(n):
            # Boundary gradient
            b_exp_sum = np.sum(exp_b[i, :])
            if b_exp_sum > 0:
                # d(min_boundary_soft[i])/dx[i]
                d_dxi = -(1/alpha) * (exp_b[i, 0] * (-alpha) + exp_b[i, 1] * alpha) / b_exp_sum
                # d(min_boundary_soft[i])/dy[i]
                d_dyi = -(1/alpha) * (exp_b[i, 2] * (-alpha) + exp_b[i, 3] * alpha) / b_exp_sum

                # Distance gradient
                # min_dist_soft[i] = -log_sum[i] / alpha
                # d(min_dist_soft[i])/dx[i] involves all j != i
                d_exp_sum = np.sum(exp_scaled[i, :])
                if d_exp_sum > 0:
                    # For each j != i, d(dist_matrix[i,j])/dx[i] = dx[i,j] / dist_matrix[i,j]
                    # d(scaled[i,j])/dx[i] = -alpha * dx[i,j] / dist_matrix[i,j]
                    # d(log_sum[i])/dx[i] = sum_j exp_scaled[i,j] * d(scaled[i,j])/dx[i] / d_exp_sum

                    d_dist_dxi = np.zeros(n)
                    d_dist_dyi = np.zeros(n)
                    for j in range(n):
                        if i == j:
                            continue
                        ddist = dist_matrix[i, j]
                        if ddist > 1e-10:
                            d_dist_dxi[j] = dx[i, j] / ddist
                            d_dist_dyi[j] = dy[i, j] / ddist

                    d_dxi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dxi)) / d_exp_sum
                    d_dyi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dyi)) / d_exp_sum

                # Combine boundary and distance via soft-min
                # radii_soft[i] = -log_sum_ab[i] / alpha
                # where scaled_ab[i,0] = -alpha * a[i], scaled_ab[i,1] = -alpha * b[i]
                # d(radii_soft[i])/dx[i] = -(1/alpha) * d(log_sum_ab[i])/dx[i]
                # d(log_sum_ab[i])/dx[i] = sum_k exp_ab[i,k] * d(scaled_ab[i,k])/dx[i] / sum_k exp_ab[i,k]
                # d(scaled_ab[i,0])/dx[i] = -alpha * d(a[i])/dx[i] = -alpha * d_dxi
                # d(scaled_ab[i,1])/dx[i] = -alpha * d(b[i])/dx[i] = -alpha * (1/2) * d(min_dist_soft[i])/dx[i]

                ab_exp_sum = np.sum(exp_ab[i, :])
                if ab_exp_sum > 0:
                    # d(radii_soft[i])/dx[i]
                    dr_dxi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dxi) + exp_ab[i, 1] * (-alpha * 0.5 * (d_dxi))) / ab_exp_sum
                    dr_dyi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dyi) + exp_ab[i, 1] * (-alpha * 0.5 * (d_dyi))) / ab_exp_sum

                    if radii_soft[i] > 0:
                        grad[2*i] = -dr_dxi
                        grad[2*i+1] = -dr_dyi
            else:
                # If no boundary terms contribute, still need distance gradient
                d_exp_sum = np.sum(exp_scaled[i, :])
                if d_exp_sum > 0:
                    d_dist_dxi = np.zeros(n)
                    d_dist_dyi = np.zeros(n)
                    for j in range(n):
                        if i == j:
                            continue
                        ddist = dist_matrix[i, j]
                        if ddist > 1e-10:
                            d_dist_dxi[j] = dx[i, j] / ddist
                            d_dist_dyi[j] = dy[i, j] / ddist

                    d_dxi = -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dxi)) / d_exp_sum
                    d_dyi = -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dyi)) / d_exp_sum

                    ab_exp_sum = np.sum(exp_ab[i, :])
                    if ab_exp_sum > 0:
                        dr_dxi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * 0) + exp_ab[i, 1] * (-alpha * 0.5 * d_dxi)) / ab_exp_sum
                        dr_dyi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * 0) + exp_ab[i, 1] * (-alpha * 0.5 * d_dyi)) / ab_exp_sum

                        if radii_soft[i] > 0:
                            grad[2*i] = -dr_dxi
                            grad[2*i+1] = -dr_dyi

        return f, grad

    def push_apart(params, iterations=50):
        """Push overlapping circles apart to improve validity."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        radii = compute_radii(x, y)

        for _ in range(iterations):
            # Find pairs that are too close
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        push = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        x[i] -= push * nx
                        y[i] -= push * ny
                        x[j] += push * nx
                        y[j] += push * ny
                        moved = True
            if not moved:
                break

        # Clip to bounds
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        new_params = np.zeros(2 * N)
        new_params[0::2] = x
        new_params[1::2] = y
        return new_params

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
        """Hexagonal close-packing initialization."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        s = 1.0 / np.sqrt(2 * N)
        for row in range(8):
            for col in range(8):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.05, 0.95)
                y0[idx] = np.clip(y0[idx], 0.05, 0.95)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def random_init():
        """Random initialization with some spacing."""
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Try multiple initializations
    inits = [grid_init(), hex_init()]
    np.random.seed(42)
    for _ in range(3):
        inits.append(random_init())

    for init_params in inits:
        if time.time() > deadline - 60:
            break
        obj_val = objective(init_params)
        if obj_val < best_obj:
            best_params = init_params.copy()
            best_obj = obj_val

    # Multi-stage optimization with increasing alpha (smoothing sharpness)
    # Stage 1: Coarse search with low alpha
    # Stage 2: Medium search
    # Stage 3: Fine search
    # Stage 4: Exact Powell polish

    alphas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for i, alpha in enumerate(alphas):
        if time.time() > deadline - 30:
            break
            
        # For each alpha, try a few restarts from best known and random
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                params_start = best_params.copy()
            elif restart == 1:
                # Jittered best
                params_start = best_params + np.random.randn(2*N) * 0.02
                params_start = np.clip(params_start, 0.01, 0.99)
            else:
                # More aggressive jitter
                params_start = best_params + np.random.randn(2*N) * 0.05
                params_start = np.clip(params_start, 0.01, 0.99)
            
            try:
                # Use smooth objective with gradient for L-BFGS-B
                # Lower ftol for earlier stages, tighter for later
                ftol = 1e-10 if i < 2 else 1e-12
                res = minimize(
                    lambda p: smooth_objective_grad(p, alpha=alpha), 
                    params_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    jac=True,
                    options={'maxiter': 3000, 'ftol': ftol, 'gtol': 1e-7}
                )
                # Evaluate with the true (non-smooth) objective
                true_obj = objective(res.x)
                if true_obj < best_obj:
                    # Apply push_apart to clean up any near-overlaps
                    res_x = push_apart(res.x, iterations=30)
                    pa_obj = objective(res_x)
                    if pa_obj < best_obj:
                        best_params = res_x
                        best_obj = pa_obj
                    else:
                        best_params = res.x
                        best_obj = true_obj
            except Exception:
                pass

    # Phase 3: Powell refinement from best smooth result
    if time.time() < deadline - 25:
        try:
            # First do a push-apart to clean up
            cleaned = push_apart(best_params, iterations=50)
            clean_obj = objective(cleaned)
            if clean_obj < best_obj:
                best_params = cleaned
                best_obj = clean_obj

            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 8000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            # Another push-apart
            cleaned = push_apart(best_params, iterations=100)
            clean_obj = objective(cleaned)
            if clean_obj < best_obj:
                best_params = cleaned
                best_obj = clean_obj

            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 15000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 5: Final cleanup and verification
    if time.time() < deadline - 3:
        try:
            final = push_apart(best_params, iterations=200)
            final_obj = objective(final)
            if final_obj < best_obj:
                best_params = final
                best_obj = final_obj
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.99999

    # Ensure circles are within bounds
    for i in range(N):
        if x[i] < radii[i]:
            x[i] = radii[i]
        if x[i] > 1 - radii[i]:
            x[i] = 1 - radii[i]
        if y[i] < radii[i]:
            y[i] = radii[i]
        if y[i] > 1 - radii[i]:
            y[i] = 1 - radii[i]

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
