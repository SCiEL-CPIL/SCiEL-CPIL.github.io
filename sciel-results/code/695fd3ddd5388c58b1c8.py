"""Circle packing in perimeter-4 rectangle.
Structural change: SLSQP with hard constraints + multi-init + aspect search."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 135.0  # safety margin before 150s kill


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


def _separate_circles(circles, W, H, max_iter=3000, tol=1e-12):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
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

        # Re-clamp after separation, shrinking radius if circle doesn't fit
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


def _slsqp_objective(params, W, H):
    """Objective: minimize -sum(r) = maximize sum(r)."""
    r = params[2::3]
    return -np.sum(r)


def _slsqp_gradient(params, W, H):
    """Gradient of objective."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    return grad


def _slsqp_constraints(W, H):
    """Build constraint functions for SLSQP."""
    n = N
    constraints = []

    # Boundary constraints: x_i - r_i >= 0
    def c_xmin(params):
        return params[0::3] - params[2::3]
    constraints.append({'type': 'ineq', 'fun': c_xmin})

    # Boundary constraints: y_i - r_i >= 0
    def c_ymin(params):
        return params[1::3] - params[2::3]
    constraints.append({'type': 'ineq', 'fun': c_ymin})

    # Boundary constraints: W - x_i - r_i >= 0
    def c_xmax(params):
        return W - params[0::3] - params[2::3]
    constraints.append({'type': 'ineq', 'fun': c_xmax})

    # Boundary constraints: H - y_i - r_i >= 0
    def c_ymax(params):
        return H - params[1::3] - params[2::3]
    constraints.append({'type': 'ineq', 'fun': c_ymax})

    # Non-negativity: r_i >= 0 (handled by bounds)

    # Overlap constraints: dist_ij - r_i - r_j >= 0
    iu, ju = np.triu_indices(n, k=1)
    for idx, (i, j) in enumerate(zip(iu, ju)):
        def overlap_c(params, i=i, j=j):
            dx = params[3 * i] - params[3 * j]
            dy = params[3 * i + 1] - params[3 * j + 1]
            dist = np.sqrt(dx * dx + dy * dy)
            return dist - params[3 * i + 2] - params[3 * j + 2]
        constraints.append({'type': 'ineq', 'fun': overlap_c})

    return constraints


def _slsqp_optimize(circles, W, H, max_iter=2000):
    """Optimize using SLSQP with hard constraints."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    constraints = _slsqp_constraints(W, H)

    result = minimize(
        _slsqp_objective, params, args=(W, H),
        method='SLSQP',
        jac=_slsqp_gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-12}
    )
    return result.x.reshape(N, 3)


def _penalty_objective(params, W, H):
    """Penalty-based objective for L-BFGS-B as fallback."""
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    eps = 1e-4
    boundary_weight = 200.0

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

    # Overlap penalties
    overlap_weight = 200.0
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


def _lbgfbs_optimize(circles, W, H, max_iter=3000):
    """Fallback L-BFGS-B with penalty."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _penalty_objective, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish using SLSQP."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    constraints = _slsqp_constraints(W, H)

    # Pass 1: coarse
    result = minimize(
        _slsqp_objective, params, args=(W, H),
        method='SLSQP', jac=_slsqp_gradient,
        bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter // 3, 'ftol': 1e-10}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _slsqp_objective, params, args=(W, H),
        method='SLSQP', jac=_slsqp_gradient,
        bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter // 3, 'ftol': 1e-12}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _slsqp_objective, params, args=(W, H),
        method='SLSQP', jac=_slsqp_gradient,
        bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter // 3, 'ftol': 1e-14}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create hexagonal lattice initial configuration."""
    circles = []
    best_circles = None
    best_count = 0

    for trial_r in np.linspace(0.06, 0.28, 25):
        dx = 2 * trial_r * 1.002
        dy = 2 * trial_r * np.sqrt(3) / 2 * 1.002

        row = 0
        y_pos = trial_r + dy / 2
        while y_pos + trial_r <= H + 1e-9 and len(circles) < N:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x_pos = trial_r + offset
            row += 1
            while x_pos + trial_r <= W + 1e-9 and len(circles) < N:
                circles.append([x_pos, y_pos, trial_r])
                x_pos += dx
            y_pos += dy

        if len(circles) > best_count:
            best_count = len(circles)
            best_circles = circles.copy()
        circles = []

    if best_circles is None or len(best_circles) == 0:
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])

    if len(circles) > N:
        circles = circles[:N]

    while len(circles) < N:
        best_x, best_y, best_r = W / 2, H / 2, 0.01
        for _ in range(50):
            tx = np.random.uniform(0.05, W - 0.05)
            ty = np.random.uniform(0.05, H - 0.05)
            min_dist = min(W, H)
            for c in circles:
                d = np.sqrt((tx - c[0]) ** 2 + (ty - c[1]) ** 2)
                min_dist = min(min_dist, d - c[2])
            min_dist = min(min_dist, tx, W - tx, ty, H - ty)
            if min_dist > best_r:
                best_r = min_dist * 0.9
                best_x, best_y = tx, ty
        best_r = max(0.01, best_r)
        circles.append([best_x, best_y, best_r])

    return np.array(circles[:N])


def _grid_initialization(W, H):
    """Grid-based initialization."""
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
    """Random initialization."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _best_for_aspect(W, H, time_budget=20.0):
    """Find best configuration for given W, H."""
    start = time.time()

    # Try hexagonal init
    circ_hex = _hexagonal_initialization(W, H)
    best_val = -np.sum(circ_hex[:, 2])
    best_circles = circ_hex.copy()

    # Try grid init
    circ_grid = _grid_initialization(W, H)
    if -np.sum(circ_grid[:, 2]) > best_val:
        best_val = -np.sum(circ_grid[:, 2])
        best_circles = circ_grid.copy()

    # SLSQP on hex init
    if time.time() - start < time_budget - 5:
        circ_s = _slsqp_optimize(circ_hex, W, H, max_iter=1500)
        circ_s[:, 0] = np.clip(circ_s[:, 0], circ_s[:, 2], W - circ_s[:, 2])
        circ_s[:, 1] = np.clip(circ_s[:, 1], circ_s[:, 2], H - circ_s[:, 2])
        circ_s[:, 2] = np.clip(circ_s[:, 2], 0.001, min(W, H) / 2)
        circ_s, valid_s = _separate_circles(circ_s, W, H)
        if valid_s:
            s = np.sum(circ_s[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circ_s.copy()

    # SLSQP on grid init
    if time.time() - start < time_budget - 5:
        circ_s2 = _slsqp_optimize(circ_grid, W, H, max_iter=1500)
        circ_s2[:, 0] = np.clip(circ_s2[:, 0], circ_s2[:, 2], W - circ_s2[:, 2])
        circ_s2[:, 1] = np.clip(circ_s2[:, 1], circ_s2[:, 2], H - circ_s2[:, 2])
        circ_s2[:, 2] = np.clip(circ_s2[:, 2], 0.001, min(W, H) / 2)
        circ_s2, valid_s2 = _separate_circles(circ_s2, W, H)
        if valid_s2:
            s = np.sum(circ_s2[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circ_s2.copy()

    # Polish
    if time.time() - start < time_budget - 3:
        polished = _polish(best_circles, W, H, max_iter=4000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_p = _separate_circles(polished, W, H)
        if valid_p:
            s = np.sum(polished[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = polished.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios to try (W/H ratio)
    aspect_ratios = [0.85, 0.92, 1.0, 1.08, 1.15, 1.35, 1.6]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

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

        # Random restarts if time permits
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 8))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _slsqp_optimize(circ_rand, W, H, max_iter=800)
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

    # Final high-precision polish
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 4:
        polished2 = _polish(best_result, best_W, best_H, max_iter=4000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=1000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Fallback: safe grid in square
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

    # Ensure validity; fall back to safe grid if needed
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
