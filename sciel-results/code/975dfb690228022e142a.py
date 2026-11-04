"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
    """Directly resolve overlaps and boundary violations using pairwise repulsion."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
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
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _hexagonal_layout(W, H, r):
    """
    Generate a hexagonal lattice of circles with radius r in rectangle W x H.
    Returns array of shape (n, 2) with centers.
    """
    centers = []
    dx = 2 * r * 1.002
    dy = 2 * r * np.sqrt(3) / 2 * 1.002

    row = 0
    y_pos = r + dy / 2
    while y_pos + r <= H + 1e-9:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x_pos = r + offset
        row += 1
        while x_pos + r <= W + 1e-9:
            centers.append([x_pos, y_pos])
            x_pos += dx
        y_pos += dy

    return np.array(centers[:N]) if len(centers) >= N else np.array(centers)


def _slsqp_fixed_radius(circles, W, H, r_fixed, max_iter=2000):
    """
    Optimize only center coordinates with fixed radius using SLSQP.
    circles: (N, 3) array, but only x,y are optimized.
    """
    n = N
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()

    def objective(params):
        x = params[0:n]
        y = params[n:2*n]
        # Minimize negative sum of radii (constant here) + penalty for violations
        # Actually, with fixed radius, we just need to satisfy constraints.
        # Objective: minimize 0 (or a small penalty for being out of bounds)
        # SLSQP will handle constraints, so objective can be constant.
        return 0.0

    def objective_grad(params):
        return np.zeros(2 * n)

    # Constraints: non-overlap: dist_ij >= r_i + r_j
    # Boundary: x_i >= r, x_i <= W-r, y_i >= r, y_i <= H-r

    def _make_overlap_constraint(i, j):
        def con(params):
            x = params[0:n]
            y = params[n:2*n]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - 2 * r_fixed

        def con_grad(params):
            x = params[0:n]
            y = params[n:2*n]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            g = np.zeros(2 * n)
            g[i] = dx / dist
            g[n + i] = dy / dist
            g[j] = -dx / dist
            g[n + j] = -dy / dist
            return g

        return {'type': 'ineq', 'fun': con, 'jac': con_grad}

    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(_make_overlap_constraint(i, j))

    # Boundary constraints
    for i in range(n):
        # x_i >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[i] - r_fixed,
            'jac': lambda p, i=i: np.concatenate([[1 if k == i else 0 for k in range(2*n)]])
        })
        # x_i <= W - r
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: (W - r_fixed) - p[i],
            'jac': lambda p, i=i: np.concatenate([[-1 if k == i else 0 for k in range(2*n)]])
        })
        # y_i >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[n + i] - r_fixed,
            'jac': lambda p, i=i: np.concatenate([[0]*n + [1 if k == i else 0 for k in range(n)]])
        })
        # y_i <= H - r
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: (H - r_fixed) - p[n + i],
            'jac': lambda p, i=i: np.concatenate([[0]*n + [-1 if k == i else 0 for k in range(n)]])
        })

    x0_full = np.concatenate([x0, y0])
    bounds = [(r_fixed * 0.5, W - r_fixed * 0.5)] * n + [(r_fixed * 0.5, H - r_fixed * 0.5)] * n

    result = minimize(
        objective, x0_full, jac=objective_grad,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-12}
    )

    if result.success:
        x_opt = result.x[0:n]
        y_opt = result.x[n:2*n]
        new_circles = np.column_stack([x_opt, y_opt, np.full(n, r_fixed)])
        new_circles[:, 0] = np.clip(new_circles[:, 0], r_fixed, W - r_fixed)
        new_circles[:, 1] = np.clip(new_circles[:, 1], r_fixed, H - r_fixed)
        return new_circles
    return circles


def _penalty_objective(params, W, H, eps=1e-4, boundary_weight=500.0, overlap_weight=500.0):
    """
    Penalty-based objective for joint (x, y, r) optimization.
    Uses smooth max for constraints.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized
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


def _local_optimize(circles, W, H, max_iter=3000, eps=1e-4):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _penalty_objective, params, args=(W, H, eps),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Three-pass polish: coarse, medium, fine."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-4),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 3, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-5),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 3, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 150}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-6),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create a hexagonal lattice initial configuration."""
    circles = []
    best_circles = None
    best_count = 0

    for trial_r in np.linspace(0.08, 0.25, 25):
        centers = _hexagonal_layout(W, H, trial_r)
        if len(centers) >= N:
            circles = np.column_stack([centers[:N], np.full(N, trial_r)])
            if trial_r > (best_circles[:, 2][0] if best_circles is not None else 0):
                best_circles = circles.copy()

    if best_circles is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])
        best_circles = np.array(circles[:N])

    return best_circles


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _slsqp_joint(circles, W, H, max_iter=500):
    """
    SLSQP optimization with joint (x, y, r) variables and hard constraints.
    This is slower but can find better local optima.
    """
    n = N
    params = circles.flatten().copy()

    def objective(p):
        c = p.reshape(n, 3)
        return -np.sum(c[:, 2])

    def objective_grad(p):
        g = np.zeros(3 * n)
        g[2::3] = -1.0
        return g

    constraints = []

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def _overlap_con(p, i=i, j=j):
                xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
                xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - ri - rj

            def _overlap_grad(p, i=i, j=j):
                xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
                xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dist = 1e-12
                    dx = 1e-12
                    dy = 0.0
                g = np.zeros(3 * n)
                g[3*i] = dx / dist
                g[3*i+1] = dy / dist
                g[3*i+2] = -1.0
                g[3*j] = -dx / dist
                g[3*j+1] = -dy / dist
                g[3*j+2] = -1.0
                return g

            constraints.append({'type': 'ineq', 'fun': _overlap_con, 'jac': _overlap_grad})

    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.concatenate([[1 if k == 3*i else 0 for k in range(3*n)]])
        })
        # r_i - x_i <= 0 is same as above
        # x_i + r_i <= W
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: W - p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.concatenate([[-1 if k == 3*i else 0 for k in range(3*n)]])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.concatenate([[1 if k == 3*i+1 else 0 for k in range(3*n)]])
        })
        # y_i + r_i <= H
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: H - p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.concatenate([[-1 if k == 3*i+1 else 0 for k in range(3*n)]])
        })
        # r_i >= 0 (handled by bounds)

    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    try:
        result = minimize(
            objective, params, jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = result.x.reshape(n, 3)
            new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], W - new_circles[:, 2])
            new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], H - new_circles[:, 2])
            new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, r_max)
            return new_circles
    except Exception:
        pass
    return circles


def _best_for_aspect(W, H, time_budget=20.0):
    """Find the best configuration for given W, H within time budget."""
    start = time.time()
    best_val = -np.inf
    best_circles = None

    # 1. Hexagonal initialization + local optimization
    circles_hex = _hexagonal_initialization(W, H)
    circles_opt = _local_optimize(circles_hex, W, H, max_iter=2000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid = _separate_circles(circles_opt, W, H)
    if valid:
        s = np.sum(circles_opt[:, 2])
        if s > best_val:
            best_val = s
            best_circles = circles_opt.copy()

    # 2. Grid initialization + local optimization
    if time.time() - start < time_budget * 0.5:
        circles_grid = _grid_initialization(W, H)
        circles_opt2 = _local_optimize(circles_grid, W, H, max_iter=1500)
        circles_opt2[:, 0] = np.clip(circles_opt2[:, 0], circles_opt2[:, 2], W - circles_opt2[:, 2])
        circles_opt2[:, 1] = np.clip(circles_opt2[:, 1], circles_opt2[:, 2], H - circles_opt2[:, 2])
        circles_opt2[:, 2] = np.clip(circles_opt2[:, 2], 0.001, min(W, H) / 2)
        circles_opt2, valid2 = _separate_circles(circles_opt2, W, H)
        if valid2:
            s = np.sum(circles_opt2[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circles_opt2.copy()

    # 3. SLSQP with fixed radius (if time permits)
    if best_circles is not None and time.time() - start < time_budget * 0.7:
        r_avg = np.mean(best_circles[:, 2])
        # Try a few nearby radii
        for r_trial in [r_avg * 0.95, r_avg, r_avg * 1.05]:
            if time.time() - start > time_budget * 0.85:
                break
            # Create hex layout with this radius
            centers = _hexagonal_layout(W, H, r_trial)
            if len(centers) >= N:
                circ_fixed = np.column_stack([centers[:N], np.full(N, r_trial)])
                circ_slsqp = _slsqp_fixed_radius(circ_fixed, W, H, r_trial, max_iter=1000)
                circ_slsqp, valid_slsqp = _separate_circles(circ_slsqp, W, H)
                if valid_slsqp:
                    s = np.sum(circ_slsqp[:, 2])
                    if s > best_val:
                        best_val = s
                        best_circles = circ_slsqp.copy()

    # 4. Polish the best result
    if best_circles is not None and time.time() - start < time_budget * 0.85:
        polished = _polish(best_circles, W, H, max_iter=4000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()

    # 5. SLSQP joint optimization (if time permits)
    if best_circles is not None and time.time() - start < time_budget * 0.9:
        circ_slsqp_joint = _slsqp_joint(best_circles, W, H, max_iter=300)
        circ_slsqp_joint, valid_joint = _separate_circles(circ_slsqp_joint, W, H)
        if valid_joint:
            s = np.sum(circ_slsqp_joint[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circ_slsqp_joint.copy()

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.85, 0.92, 1.0, 1.08, 1.15, 1.35, 1.6]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        # Allocate more time to promising aspect ratios
        if 0.92 <= ar <= 1.08:
            time_budget = 18.0
        elif 0.8 <= ar <= 1.4:
            time_budget = 14.0
        else:
            time_budget = 10.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_for_aspect(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Try random restarts if time permits
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 8))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=800)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    s = np.sum(circ_opt[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = circ_opt.copy()
                        best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # High-precision final polish with very tight eps
    if best_result is not None and time.time() - start_time < DEADLINE - 4:
        params = best_result.flatten()
        r_max = min(best_W, best_H) / 2
        bounds = []
        for i in range(N):
            bounds.append((1e-6, best_W - 1e-6))
            bounds.append((1e-6, best_H - 1e-6))
            bounds.append((1e-6, r_max))
        result = minimize(
            _penalty_objective, params, args=(best_W, best_H, 1e-7),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        polished2 = result.x.reshape(N, 3)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=1000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END
