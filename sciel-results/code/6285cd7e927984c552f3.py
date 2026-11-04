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


def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Uses log-sum-exp based smooth max for better gradient behavior.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    def _soft_max(v, scale):
        """Soft max of v and 0 using log-sum-exp: scale * log(1 + exp(v/scale))."""
        v_safe = np.clip(v / scale, -50, 50)
        smax = scale * np.log1p(np.exp(v_safe))
        sig = 1.0 / (1.0 + np.exp(-v_safe))
        return smax, sig

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _soft_max(v, eps)
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
    smax_vec, gsmax_vec = _soft_max(v_vec, eps)
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


def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize. Two-stage: coarse then fine."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Stage 1: coarser eps for faster convergence
    result = minimize(
        _objective_and_grad, params, args=(W, H, 5e-4, 150.0, 150.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200}
    )
    params = result.x

    # Stage 2: tighter eps for precision
    result = minimize(
        _objective_and_grad, params, args=(W, H, 1e-5, 400.0, 400.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps and increasing penalty."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive refinement: 6 passes with decreasing eps and increasing weight
    eps_schedule = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    weight_schedule = [200.0, 300.0, 400.0, 500.0, 600.0, 800.0]

    for eps, wt in zip(eps_schedule, weight_schedule):
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps, wt, wt),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // len(eps_schedule), 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
        )
        params = result.x

    return params.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=15000):
    """
    Ultra-high-precision final polish with very tight tolerances.
    Uses progressively tighter eps and higher weights.
    """
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-8, W - 1e-8))
        bounds.append((1e-8, H - 1e-8))
        bounds.append((1e-8, r_max))

    # Progressive refinement with very tight eps
    eps_schedule = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    weight_schedule = [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

    for eps, wt in zip(eps_schedule, weight_schedule):
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps, wt, wt),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // len(eps_schedule), 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500}
        )
        params = result.x

    return params.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """
    Create a hexagonal packing initial configuration.
    Hexagonal packing has ~15% better density than square packing.
    """
    best_circles = None
    best_score = -np.inf

    row_patterns = [
        [4, 5, 4, 4, 4],
        [5, 4, 5, 4, 3],
        [3, 5, 5, 5, 3],
        [4, 4, 5, 4, 4],
        [5, 5, 4, 4, 3],
        [2, 5, 5, 5, 4],
        [4, 5, 5, 4, 3],
    ]

    for pattern in row_patterns:
        if sum(pattern) != N:
            continue

        n_rows = len(pattern)
        max_cols = max(pattern)
        est_r_h = H / (2 + (n_rows - 1) * np.sqrt(3))
        est_r_w = W / (2 * max_cols)
        r_est = min(est_r_h, est_r_w) * 0.95

        circles = []
        for row_idx, n_cols in enumerate(pattern):
            y = r_est + row_idx * r_est * np.sqrt(3)
            x_offset = 0 if row_idx % 2 == 0 else r_est
            for col_idx in range(n_cols):
                row_width = (n_cols - 1) * 2 * r_est + 2 * r_est
                x_center = (W - row_width) / 2
                x = x_center + col_idx * 2 * r_est
                circles.append([x, y, r_est])

        if len(circles) != N:
            continue

        circles = np.array(circles)

        for i in range(N):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
            max_r = min(circles[i, 0], W - circles[i, 0], circles[i, 1], H - circles[i, 1])
            circles[i, 2] = min(circles[i, 2], max_r)

        circles, valid = _separate_circles(circles, W, H)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()

    if best_circles is None:
        return _grid_initialization(W, H)

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
    """Create a random initial configuration with moderate circles."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _gradient_flow(circles, W, H, dt=0.01, max_steps=5000, time_budget=10.0):
    """
    Semi-implicit Euler gradient flow with projection onto feasible set.
    """
    start = time.time()
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    r_max = min(W, H) / 2

    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for step in range(max_steps):
        if time.time() - start > time_budget:
            break

        params = np.column_stack([x, y, r]).flatten()
        obj, grad = _objective_and_grad(params, W, H, 1e-4, 200.0, 200.0)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        effective_dt = dt / (1.0 + grad_norm * 0.1)

        new_params = params - effective_dt * grad
        new_x = new_params[0::3]
        new_y = new_params[1::3]
        new_r = new_params[2::3]

        new_r = np.clip(new_r, 1e-4, r_max)
        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)

        for i in range(n):
            for j in range(i + 1, n):
                dx = new_x[i] - new_x[j]
                dy = new_y[i] - new_y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = new_r[i] + new_r[j]
                if dist_sq < min_dist ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = new_r[i] + new_r[j]
                    wi = new_r[i] / total_r if total_r > 0 else 0.5
                    wj = new_r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    new_x[i] += wi * push * dx / dist
                    new_y[i] += wi * push * dy / dist
                    new_x[j] -= wj * push * dx / dist
                    new_y[j] -= wj * push * dy / dist

        for i in range(n):
            max_r_x = min(new_x[i], W - new_x[i])
            max_r_y = min(new_y[i], H - new_y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if new_r[i] > max_r:
                new_r[i] = max_r
            new_x[i] = np.clip(new_x[i], new_r[i], W - new_r[i])
            new_y[i] = np.clip(new_y[i], new_r[i], H - new_r[i])

        val = np.sum(new_r)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r])


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    best_val = -np.inf
    best_circles = None

    inits = [
        ('grid', _grid_initialization(W, H)),
        ('hex', _hexagonal_initialization(W, H)),
    ]

    for name, circles in inits:
        if time.time() - start > time_budget - 8:
            break

        circles_opt = _local_optimize(circles, W, H, max_iter=2000)
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

        circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
        if valid_opt:
            s = np.sum(circles_opt[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circles_opt.copy()

    if best_circles is not None and time.time() - start < time_budget - 5:
        gf_circles = _gradient_flow(best_circles, W, H, dt=0.005, max_steps=3000, time_budget=time_budget - 10)
        gf_circles[:, 0] = np.clip(gf_circles[:, 0], gf_circles[:, 2], W - gf_circles[:, 2])
        gf_circles[:, 1] = np.clip(gf_circles[:, 1], gf_circles[:, 2], H - gf_circles[:, 2])
        gf_circles[:, 2] = np.clip(gf_circles[:, 2], 0.001, min(W, H) / 2)
        gf_circles, valid_gf = _separate_circles(gf_circles, W, H)
        if valid_gf:
            s = np.sum(gf_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = gf_circles.copy()

    if best_circles is not None and time.time() - start < time_budget - 3:
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

    if best_circles is not None and time.time() - start < time_budget - 1:
        hp = _high_precision_polish(best_circles, W, H, max_iter=10000)
        hp[:, 0] = np.clip(hp[:, 0], hp[:, 2], W - hp[:, 2])
        hp[:, 1] = np.clip(hp[:, 1], hp[:, 2], H - hp[:, 2])
        hp[:, 2] = np.clip(hp[:, 2], 0.001, min(W, H) / 2)
        hp, valid_hp = _separate_circles(hp, W, H)
        if valid_hp:
            s = np.sum(hp[:, 2])
            if s > best_val:
                best_val = s
                best_circles = hp.copy()

    if best_circles is None:
        best_circles = _grid_initialization(W, H)
        best_val = np.sum(best_circles[:, 2])

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    aspect_ratios = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25, 1.4, 1.6, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        if 0.9 <= ar <= 1.1:
            time_budget = 16.0
        elif 0.75 <= ar <= 1.25:
            time_budget = 12.0
        else:
            time_budget = 8.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 4:
            n_restarts = max(1, int(remaining / 3))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=2000)
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

    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        hp = _high_precision_polish(best_result, best_W, best_H, max_iter=15000)
        hp[:, 0] = np.clip(hp[:, 0], hp[:, 2], best_W - hp[:, 2])
        hp[:, 1] = np.clip(hp[:, 1], hp[:, 2], best_H - hp[:, 2])
        hp[:, 2] = np.clip(hp[:, 2], 0.001, min(best_W, best_H) / 2)
        hp, valid_final = _separate_circles(hp, best_W, best_H)
        if valid_final:
            s = np.sum(hp[:, 2])
            if s > best_score:
                best_score = s
                best_result = hp.copy()

        if best_result is not None and time.time() - start_time < DEADLINE - 2:
            hp2 = _high_precision_polish(best_result, best_W, best_H, max_iter=8000)
            hp2[:, 0] = np.clip(hp2[:, 0], hp2[:, 2], best_W - hp2[:, 2])
            hp2[:, 1] = np.clip(hp2[:, 1], hp2[:, 2], best_H - hp2[:, 2])
            hp2[:, 2] = np.clip(hp2[:, 2], 0.001, min(best_W, best_H) / 2)
            hp2, valid_final2 = _separate_circles(hp2, best_W, best_H)
            if valid_final2:
                s = np.sum(hp2[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = hp2.copy()

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
