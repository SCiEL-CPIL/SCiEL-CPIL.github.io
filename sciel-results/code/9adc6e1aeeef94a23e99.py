import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE_MARGIN = 10.0  # safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
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


def _hexagonal_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42)

    # Estimate radius based on area packing density
    r_est = np.sqrt(0.9069 * w * h / (num_circles * np.pi))

    # We want to place circles such that they don't overlap initially.
    # Let's use a grid of potential centers.
    s = 2 * r_est * 1.05

    # Generate grid points
    nx = int(np.ceil(w / s)) + 2
    ny = int(np.ceil(h / s)) + 2

    xs = np.linspace(-s, w + s, nx)
    ys = np.linspace(-s, h + s, ny)

    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if i % 2 == 1:
                y_offset = y + s / 2
            else:
                y_offset = y
            points.append([x, y_offset])

    points = np.array(points)

    # Filter points that are reasonably inside
    valid_mask = (points[:, 0] >= 0) & (points[:, 0] <= w) & \
                 (points[:, 1] >= 0) & (points[:, 1] <= h)
    points = points[valid_mask]

    if len(points) > num_circles:
        cx, cy = w/2, h/2
        dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        sorted_idx = np.argsort(dist_to_center)
        points = points[sorted_idx[:num_circles]]
    elif len(points) < num_circles:
        missing = num_circles - len(points)
        extra_points = rng.rand(missing, 2) * np.array([w, h])
        points = np.vstack([points, extra_points])

    order = rng.permutation(num_circles)
    points = points[order]

    radii = np.full(num_circles, r_est * 0.9)

    return points, radii


def _random_start(w, h, num_circles, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(0)

    pos = rng.rand(num_circles, 2) * np.array([w, h])
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r

    return pos, radii


def _pack_for_aspect(w, h, start_time, deadline):
    best_score = 0
    best_circles = None

    max_r = min(w, h) / 2.0

    initializations = []

    # 1. Hexagonal lattice start
    if time.time() < deadline - 15:
        pos_hex, rad_hex = _hexagonal_start(w, h, N, seed=42)
        circles_hex = np.column_stack([pos_hex, rad_hex])
        initializations.append(circles_hex)

        # Repair initial hex to ensure validity for baseline
        circles_hex, valid_hex = _separate_circles(circles_hex, w, h)
        if valid_hex:
            score_hex = np.sum(circles_hex[:, 2])
            if score_hex > best_score:
                best_score = score_hex
                best_circles = circles_hex.copy()

    # 2. Random restarts
    if time.time() < deadline - 10:
        for seed in [123, 456, 789, 1011, 2023, 3456, 7890, 1234]:
            if time.time() > deadline - 5:
                break
            pos_rand, rad_rand = _random_start(w, h, N, seed=seed)
            circles_rand = np.column_stack([pos_rand, rad_rand])
            initializations.append(circles_rand)

    for idx, circles_init in enumerate(initializations):
        if time.time() > deadline - 10:
            break

        try:
            # Local optimization with analytical gradients
            circles_opt = _local_optimize(circles_init, w, h, max_iter=2000)

            # Repair to ensure validity
            circles_opt, valid_opt = _separate_circles(circles_opt, w, h)

            if valid_opt:
                # Polish phase
                if time.time() < deadline - 5:
                    circles_polished = _polish(circles_opt, w, h, max_iter=4000)
                    circles_polished, valid_polished = _separate_circles(circles_polished, w, h)
                    if valid_polished:
                        circles_opt = circles_polished

                score_opt = np.sum(circles_opt[:, 2])
                if score_opt > best_score:
                    best_score = score_opt
                    best_circles = circles_opt.copy()

        except Exception:
            continue

    return best_circles, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # Safety margin

    best_global_score = 0
    best_global_result = None

    np.random.seed(42)

    # Try a range of aspect ratios
    aspect_ratios = np.linspace(0.8, 1.25, 10)

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)

        circles, score = _pack_for_aspect(w, h, start_time, deadline)

        if circles is not None and score > best_global_score:
            # Double check validity
            if _check_validity(circles, w, h):
                best_global_score = score
                best_global_result = circles.copy()

    # Fallback if no valid result found
    if best_global_result is None:
        w = 1.0
        h = 1.0
        pos, radii = _hexagonal_start(w, h, N, seed=42)
        circles = np.column_stack([pos, radii])
        circles, _ = _separate_circles(circles, w, h)
        best_global_result = circles

    return best_global_result
