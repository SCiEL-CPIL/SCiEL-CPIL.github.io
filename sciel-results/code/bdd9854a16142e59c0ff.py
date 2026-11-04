"""Circle packing in perimeter-4 rectangle.
Pack 21 circles with disjoint interiors, maximize sum of radii.
Perimeter constraint: 2*(w+h) <= 4  =>  w+h <= 2.
"""
N = 21
import time
import numpy as np
from scipy.optimize import minimize

# Safety margin before 150s deadline
DEADLINE = 140.0


def _validate(w, h, x, y, r):
    """Check all constraints; return True if valid."""
    if 2.0 * (w + h) > 4.0 + 1e-7:
        return False
    for i in range(N):
        if x[i] < r[i] - 1e-7 or x[i] > w - r[i] + 1e-7:
            return False
        if y[i] < r[i] - 1e-7 or y[i] > h - r[i] + 1e-7:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_d = r[i] + r[j]
            if dist_sq < min_d * min_d - 1e-7:
                return False
    return True


def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
    """
    Objective: minimize -sum(r) + smooth penalties for boundary and overlap violations.
    params: flat array of shape (3N,) = [x0,y0,r0, x1,y1,r1, ...]
    Returns (obj, grad).
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # Boundary: x >= r
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # Boundary: y >= r
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # Boundary: x + r <= W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # Boundary: y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # Boundary: r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties (vectorized)
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += overlap_weight * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

    grad_x_i = factor_vec * nx_ij
    grad_y_i = factor_vec * ny_ij
    grad_x_j = -factor_vec * nx_ij
    grad_y_j = -factor_vec * ny_ij

    np.add.at(grad, 3 * iu, grad_x_i)
    np.add.at(grad, 3 * iu + 1, grad_y_i)
    np.add.at(grad, 3 * iu + 2, factor_vec)
    np.add.at(grad, 3 * ju, grad_x_j)
    np.add.at(grad, 3 * ju + 1, grad_y_j)
    np.add.at(grad, 3 * ju + 2, factor_vec)

    return obj, grad


def _make_initial(w, h, seed):
    """Create a random initial configuration for N circles in [0,w]x[0,h]."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    dx = w / cols
    dy = h / rows

    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    for i in range(N):
        ci = i % cols
        ri = i // cols
        if ri >= rows:
            ri = rows - 1
        x[i] = (ci + 0.5) * dx + rng.uniform(-0.15 * dx, 0.15 * dx)
        y[i] = (ri + 0.5) * dy + rng.uniform(-0.15 * dy, 0.15 * dy)
        r[i] = 0.45 * min(dx, dy) * 0.5

    for i in range(N):
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])

    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _separate_and_clamp(params, W, H, max_iter=2000):
    """Directly resolve overlaps and boundary violations using pairwise repulsion."""
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()

    for it in range(max_iter):
        moved = False

        for i in range(n):
            if r[i] <= 1e-6:
                r[i] = 1e-6
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - 1e-12) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-6, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    params_out = np.zeros(3 * N)
    params_out[0::3] = x
    params_out[1::3] = y
    params_out[2::3] = r
    return params_out


def _optimize_single(w, h, seed):
    """Run one optimization for fixed aspect ratio with progressive refinement."""
    params = _make_initial(w, h, seed)
    
    # Pre-separation to get a feasible starting point
    params = _separate_and_clamp(params, w, h, max_iter=500)
    
    r_max = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, w - 1e-6))
        bounds.append((1e-6, h - 1e-6))
        bounds.append((1e-6, r_max))

    best_obj = _objective_and_grad(params, w, h, eps=1e-3)[0]
    best_params = params.copy()

    # Progressive refinement: coarse to fine smooth-max eps
    eps_schedule = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for eps_val in eps_schedule:
        res = minimize(
            _objective_and_grad, best_params, args=(w, h, eps_val, 200.0, 200.0),
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()

    # Final separation and clamping
    best_params = _separate_and_clamp(best_params, w, h, max_iter=1000)
    
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]

    if _validate(w, h, x, y, r):
        return np.sum(r), x, y, r
    return None


def construct():
    start_time = time.time()

    best_score = -np.inf
    best_result = None

    # Grid of aspect ratios to try
    w_values = np.linspace(0.6, 1.4, 9)

    seed_counter = 0

    for w in w_values:
        if time.time() - start_time > DEADLINE:
            break
        h = 2.0 - w

        # Try a few random seeds per aspect ratio
        n_seeds = 3
        for s in range(n_seeds):
            if time.time() - start_time > DEADLINE:
                break
            seed_counter += 1
            result = _optimize_single(w, h, seed_counter)
            if result is not None:
                score, x, y, r = result
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])

    # If no valid result found, fall back to a simple grid
    if best_result is None:
        side = 5
        r = 0.5 / side - 1e-4
        x = np.zeros(N)
        y = np.zeros(N)
        rr = np.zeros(N)
        for i in range(N):
            ci = i % side
            ri = i // side
            x[i] = (ci + 0.5) / side
            y[i] = (ri + 0.5) / side
            rr[i] = r
        best_result = np.column_stack([x, y, rr])

    return best_result
