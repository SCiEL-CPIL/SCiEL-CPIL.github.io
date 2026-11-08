"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    # Check bounds
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist - 1e-9:
                return False, 0.0
    # Check perimeter
    perim = 2 * (W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    value = np.sum(radii)
    return True, value


def _smooth_penalty(radii, x, y, W, H, beta=100.0):
    """Compute smooth penalty for violations."""
    n = len(radii)
    penalty = 0.0

    # Boundary violations
    for i in range(n):
        pen = max(0.0, radii[i] - x[i])
        penalty += pen * pen * beta
        pen = max(0.0, x[i] + radii[i] - W)
        penalty += pen * pen * beta
        pen = max(0.0, radii[i] - y[i])
        penalty += pen * pen * beta
        pen = max(0.0, y[i] + radii[i] - H)
        penalty += pen * pen * beta

    # Pairwise distance violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            pen = max(0.0, min_dist - dist)
            penalty += pen * pen * beta

    # Perimeter violation
    perim = 2 * (W + H)
    pen = max(0.0, perim - 4.0)
    penalty += pen * pen * beta * 10

    return penalty


def _objective(params, W, H, beta=100.0):
    """Objective function: minimize negative sum of radii + penalty."""
    n = N
    x = params[:n]
    y = params[n:2 * n]
    radii = params[2 * n:]

    # Radii must be positive
    radii = np.maximum(radii, 0.005)

    penalty = _smooth_penalty(radii, x, y, W, H, beta)
    value = np.sum(radii)

    return penalty - value


def _seed_grid(W, H):
    """Create a 5x5 grid of circles in a W x H rectangle."""
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side * W
            cy = (j + 0.5) / side * H
            r = 0.5 / side * min(W, H) - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:N])


def _optimize_scipy(initial, W, H, deadline, beta_start=50.0, beta_end=500.0):
    """Optimize using scipy with smooth penalty, increasing beta over time."""
    n = N
    x0 = initial[:, 0].copy()
    y0 = initial[:, 1].copy()
    r0 = initial[:, 2].copy()

    params0 = np.concatenate([x0, y0, r0])

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x
    for i in range(n):
        bounds.append((0.0, H))  # y
    for i in range(n):
        bounds.append((0.005, min(W, H) / 2))  # r

    best_params = params0.copy()
    _, best_val = _validate_and_compute(r0, x0, y0, W, H)

    # Anneal beta from low to high
    n_iterations = 6
    for iter in range(n_iterations):
        if time.time() > deadline:
            break

        beta = beta_start * (beta_end / beta_start) ** (iter / max(1, n_iterations - 1))

        # Try different initial points
        for restart in range(3):
            if time.time() > deadline:
                break

            if restart == 0:
                x_init = x0.copy()
                y_init = y0.copy()
                r_init = r0.copy()
            else:
                x_init = x0 + np.random.uniform(-0.05, 0.05, n)
                y_init = y0 + np.random.uniform(-0.05, 0.05, n)
                r_init = np.maximum(r0 + np.random.uniform(-0.02, 0.02, n), 0.005)

            params_init = np.concatenate([x_init, y_init, r_init])

            try:
                result = minimize(
                    _objective,
                    params_init,
                    args=(W, H, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if result.fun < best_val * 0.5 or (result.success and result.fun < 0):
                    params = result.x
                    x_new = params[:n]
                    y_new = params[n:2 * n]
                    r_new = np.maximum(params[2 * n:], 0.005)

                    # Clamp to valid range
                    for i in range(n):
                        r_new[i] = min(r_new[i], W / 2, H / 2)
                        x_new[i] = np.clip(x_new[i], r_new[i], W - r_new[i])
                        y_new[i] = np.clip(y_new[i], r_new[i], H - r_new[i])

                    is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
                    if is_valid and val > best_val:
                        best_val = val
                        best_params = np.concatenate([x_new, y_new, r_new])
                        x0 = x_new.copy()
                        y0 = y_new.copy()
                        r0 = r_new.copy()

            except Exception:
                pass

    x_best = best_params[:n]
    y_best = best_params[n:2 * n]
    r_best = np.maximum(best_params[2 * n:], 0.005)

    return np.column_stack([x_best, y_best, r_best]), best_val


def _local_refine(circles, W, H, deadline, max_iter=1000):
    """Fine-tune with local search on individual circles."""
    n = len(circles)
    best_x = circles[:, 0].copy()
    best_y = circles[:, 1].copy()
    best_r = circles[:, 2].copy()
    _, best_val = _validate_and_compute(best_r, best_x, best_y, W, H)

    step_size = 0.01
    no_improve = 0
    for iter in range(max_iter):
        if time.time() > deadline:
            break

        idx = np.random.randint(n)
        x_new = best_x.copy()
        y_new = best_y.copy()
        r_new = best_r.copy()

        x_new[idx] += np.random.uniform(-step_size, step_size)
        y_new[idx] += np.random.uniform(-step_size, step_size)
        r_new[idx] = max(0.005, r_new[idx] + np.random.uniform(-0.003, 0.003))

        # Clamp
        r_new[idx] = min(r_new[idx], W / 2, H / 2)
        x_new[idx] = np.clip(x_new[idx], r_new[idx], W - r_new[idx])
        y_new[idx] = np.clip(y_new[idx], r_new[idx], H - r_new[idx])

        is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
        if is_valid and val > best_val:
            best_x = x_new
            best_y = y_new
            best_r = r_new
            best_val = val
            step_size *= 0.995
            no_improve = 0
        else:
            step_size = min(0.03, step_size * 1.01)
            no_improve += 1
            if no_improve > 200:
                break

    return np.column_stack([best_x, best_y, best_r]), best_val


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    # Try different rectangle aspect ratios (W + H = 2 for perimeter 4)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4286, 0.625, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3333]

    best_result = None
    best_val = 0.0

    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W

        if H <= 0.05 or W <= 0.05:
            continue

        # Start with grid
        initial = _seed_grid(W, H)

        # Scipy optimization
        result, val = _optimize_scipy(initial, W, H, deadline)

        # Local refinement
        result, val = _local_refine(result, W, H, deadline)

        if val > best_val:
            best_val = val
            best_result = result.copy()

    # Final local refinement on the best result
    if best_result is not None:
        # Determine W and H from the best result
        # We need to track which W/H gave the best result
        # For now, just do one more polish pass with the best dimensions
        # Re-derive from best_result: find tightest bounding box
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        # Use original W, H that produced this result
        # We'll just try a few nearby aspect ratios for final polish
        for W in [1.0, 0.8, 1.25, 0.7, 1.4286]:
            if time.time() > deadline:
                break
            H = 2.0 - W
            if H <= 0.05:
                continue
            result, val = _local_refine(best_result, W, H, deadline, max_iter=300)
            if val > best_val:
                best_val = val
                best_result = result.copy()

    if best_result is None:
        # Fallback: 1x1 square grid
        initial = _seed_grid(1.0, 1.0)
        best_result = initial

    return best_result
# EVOLVE-BLOCK-END
