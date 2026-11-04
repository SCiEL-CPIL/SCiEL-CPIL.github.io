"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 135.0  # seconds, safety margin before 150s kill


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
    n_passes = len(eps_schedule)
    iter_per_pass = max(400, max_iter // n_passes)

    for p in range(n_passes):
        eps_val = eps_schedule[p]
        w_val = weight_schedule[p]
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_val, w_val, w_val),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iter_per_pass, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=12000):
    """Ultra-tight final polish for the best candidate with very small eps."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Three very tight passes with tiny eps
    for eps_val, w_val in [(1e-5, 600.0), (1e-6, 800.0), (5e-7, 1000.0)]:
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_val, w_val, w_val),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // 3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 400}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """
    Create a hexagonal lattice initial configuration.
    """
    circles = []
    best_circles = None
    best_count = 0

    for trial_r in np.linspace(0.08, 0.25, 20):
        dx = 2 * trial_r * 1.001
        dy = 2 * trial_r * np.sqrt(3) / 2 * 1.001

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
        best_x, best_y, best_r = W/2, H/2, 0.01
        for _ in range(50):
            tx = np.random.uniform(0.05, W - 0.05)
            ty = np.random.uniform(0.05, H - 0.05)
            min_dist = min(W, H)
            for c in circles:
                d = np.sqrt((tx - c[0])**2 + (ty - c[1])**2)
                min_dist = min(min_dist, d - c[2])
            min_dist = min(min_dist, tx, W - tx, ty, H - ty)
            if min_dist > best_r:
                best_r = min_dist * 0.9
                best_x, best_y = tx, ty
        best_r = max(0.01, best_r)
        circles.append([best_x, best_y, best_r])

    return np.array(circles[:N])


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


def _force_directed_layout(circles, W, H, max_iter=5000):
    """
    Force-directed layout to improve packing.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    target_r = np.sqrt(W * H / N) * 0.5

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dx = np.random.uniform(-1e-6, 1e-6)
                    dy = np.random.uniform(-1e-6, 1e-6)
                    dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 50.0
            if x[i] + r[i] > W:
                fx[i] -= (x[i] + r[i] - W) * 50.0
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 50.0
            if y[i] + r[i] > H:
                fy[i] -= (y[i] + r[i] - H) * 50.0

        damping = 0.5
        for i in range(n):
            x[i] += damping * fx[i] * 0.01
            y[i] += damping * fy[i] * 0.01
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            min_space = min(x[i] - r[i], W - x[i] - r[i], y[i] - r[i], H - y[i] - r[i])
            for j in range(n):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_space = min(min_space, dist - r[j])
            if min_space > r[i] + 0.001:
                r[i] = min(target_r, r[i] + 0.001)

    result = np.column_stack([x, y, r])
    return result


def _pressure_field_optimization(W, H, time_budget=15.0):
    """
    1D pressure field optimization: fix hexagonal topology, optimize scale factor.
    Radii are determined by minimum clearance to neighbors and walls.
    """
    start = time.time()
    best_score = -np.inf
    best_circles = None

    # Generate hexagonal topology with variable spacing
    for spacing_mult in np.linspace(0.95, 1.15, 15):
        if time.time() - start > time_budget - 2:
            break

        circles = []
        trial_r = 0.15
        dx = 2 * trial_r * spacing_mult
        dy = 2 * trial_r * np.sqrt(3) / 2 * spacing_mult

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

        if len(circles) < N:
            # Fill remaining with smaller circles in gaps
            while len(circles) < N:
                best_x, best_y, best_r = W/2, H/2, 0.01
                for _ in range(100):
                    tx = np.random.uniform(0.02, W - 0.02)
                    ty = np.random.uniform(0.02, H - 0.02)
                    min_dist = min(W, H)
                    for c in circles:
                        d = np.sqrt((tx - c[0])**2 + (ty - c[1])**2)
                        min_dist = min(min_dist, d - c[2])
                    min_dist = min(min_dist, tx, W - tx, ty, H - ty)
                    if min_dist > best_r:
                        best_r = min_dist * 0.85
                        best_x, best_y = tx, ty
                best_r = max(0.005, best_r)
                circles.append([best_x, best_y, best_r])

        circles = np.array(circles[:N])

        # Now optimize: for each circle, compute max radius given topology
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()

        # Iteratively compute max feasible radius
        for _ in range(50):
            for i in range(N):
                max_r = min(x[i], W - x[i], y[i], H - y[i])
                for j in range(N):
                    if i != j:
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        max_r = min(max_r, dist - r[j])
                r[i] = max(1e-4, max_r)

        circles[:, 2] = r
        circles, valid = _separate_circles(circles, W, H, max_iter=2000)
        if valid:
            s = np.sum(circles[:, 2])
            if s > best_score:
                best_score = s
                best_circles = circles.copy()

    if best_circles is not None:
        # Polish the best pressure-field result
        if time.time() - start < time_budget - 3:
            polished = _polish(best_circles, W, H, max_iter=4000)
            polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
            polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
            polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
            polished, valid_polish = _separate_circles(polished, W, H)
            if valid_polish:
                s = np.sum(polished[:, 2])
                if s > best_score:
                    best_score = s
                    best_circles = polished.copy()

    return best_circles, best_score


def _global_aspect_ratio_search(time_budget=25.0):
    """
    Derivative-free global search over aspect ratio.
    For each aspect ratio candidate, run a quick local optimization.
    """
    start = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    def _eval_aspect_ratio(ar):
        elapsed = time.time() - start
        if elapsed > time_budget - 5:
            return 0.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Quick local optimization
        circles_hex = _hexagonal_initialization(W, H)
        circles_opt = _local_optimize(circles_hex, W, H, max_iter=1500)
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
        circles_opt, valid = _separate_circles(circles_opt, W, H)
        if valid:
            return np.sum(circles_opt[:, 2])
        return 0.0

    # Coarse scan over aspect ratio
    ars = np.linspace(0.85, 1.20, 15)
    best_ar = 1.0
    best_ar_score = -np.inf
    for ar in ars:
        if time.time() - start > time_budget - 10:
            break
        s = _eval_aspect_ratio(ar)
        if s > best_ar_score:
            best_ar_score = s
            best_ar = ar

    # Refine around best_ar
    fine_ars = np.linspace(best_ar - 0.05, best_ar + 0.05, 9)
    for ar in fine_ars:
        if time.time() - start > time_budget - 8:
            break
        if ar < 0.80 or ar > 1.30:
            continue
        s = _eval_aspect_ratio(ar)
        if s > best_ar_score:
            best_ar_score = s
            best_ar = ar

    # Full optimization at the found aspect ratio and nearby points
    candidate_ars = [best_ar]
    for delta in [-0.03, -0.015, 0.0, 0.015, 0.03]:
        candidate_ars.append(best_ar + delta)
    candidate_ars.extend([0.95, 1.0, 1.05])

    for ar in candidate_ars:
        if time.time() - start > time_budget - 5:
            break
        if ar < 0.75 or ar > 1.35:
            continue

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Full optimization at this aspect ratio
        circles_hex = _hexagonal_initialization(W, H)
        circles_opt = _local_optimize(circles_hex, W, H, max_iter=3000)
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
        circles_opt, valid = _separate_circles(circles_opt, W, H)
        if valid:
            s = np.sum(circles_opt[:, 2])
            if s > best_score:
                best_score = s
                best_result = circles_opt.copy()
                best_W, best_H = W, H

    # Also try pressure field at the best aspect ratio
    if time.time() - start < time_budget - 8:
        W = 2.0 * best_ar / (best_ar + 1.0)
        H = 2.0 / (best_ar + 1.0)
        pf_circles, pf_score = _pressure_field_optimization(W, H, time_budget=10.0)
        if pf_circles is not None and pf_score > best_score:
            best_score = pf_score
            best_result = pf_circles.copy()
            best_W, best_H = W, H

    return best_result, best_score, best_W, best_H


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()

    # Try hexagonal initialization first
    circles_hex = _hexagonal_initialization(W, H)
    best_val = -np.sum(circles_hex[:, 2])
    best_circles = circles_hex.copy()

    # Also try grid initialization
    circles_grid = _grid_initialization(W, H)
    if -np.sum(circles_grid[:, 2]) > best_val:
        best_val = -np.sum(circles_grid[:, 2])
        best_circles = circles_grid.copy()

    # Try force-directed layout
    circles_fd = _force_directed_layout(circles_hex, W, H, max_iter=3000)
    circles_fd[:, 0] = np.clip(circles_fd[:, 0], circles_fd[:, 2], W - circles_fd[:, 2])
    circles_fd[:, 1] = np.clip(circles_fd[:, 1], circles_fd[:, 2], H - circles_fd[:, 2])
    circles_fd[:, 2] = np.clip(circles_fd[:, 2], 0.001, min(W, H) / 2)
    circles_fd, valid_fd = _separate_circles(circles_fd, W, H)
    if valid_fd:
        s = np.sum(circles_fd[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_fd.copy()

    # Try local optimization with L-BFGS-B on hex init
    circles_opt = _local_optimize(circles_hex, W, H, max_iter=3000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_opt.copy()

    # Try pressure field optimization
    if time.time() - start < time_budget - 8:
        pf_circles, pf_score = _pressure_field_optimization(W, H, time_budget=time_budget - 10)
        if pf_circles is not None and pf_score > -best_val:
            best_val = -pf_score
            best_circles = pf_circles.copy()

    # Try local optimization on grid init if time permits
    if time.time() - start < time_budget - 10:
        circles_opt2 = _local_optimize(circles_grid, W, H, max_iter=2000)
        circles_opt2[:, 0] = np.clip(circles_opt2[:, 0], circles_opt2[:, 2], W - circles_opt2[:, 2])
        circles_opt2[:, 1] = np.clip(circles_opt2[:, 1], circles_opt2[:, 2], H - circles_opt2[:, 2])
        circles_opt2[:, 2] = np.clip(circles_opt2[:, 2], 0.001, min(W, H) / 2)
        circles_opt2, valid_opt2 = _separate_circles(circles_opt2, W, H)
        if valid_opt2:
            s = np.sum(circles_opt2[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circles_opt2.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 3:
        polished = _polish(best_circles, W, H, max_iter=6000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
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

    rng = np.random.default_rng(42)

    # Phase 1: Global aspect ratio search
    global_result, global_score, global_W, global_H = _global_aspect_ratio_search(time_budget=30.0)
    if global_result is not None and global_score > best_score:
        best_score = global_score
        best_result = global_result.copy()
        best_W, best_H = global_W, global_H

    # Phase 2: Try specific aspect ratios with full optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        aspect_ratios = [0.92, 0.95, 0.97, 1.0, 1.03, 1.06, 1.10]
        for ar in aspect_ratios:
            if time.time() - start_time > DEADLINE - 15:
                break

            W = 2.0 * ar / (ar + 1.0)
            H = 2.0 / (ar + 1.0)

            # Skip if we already tried this aspect ratio in phase 1
            if abs(W - best_W) < 0.01 and abs(H - best_H) < 0.01:
                continue

            time_budget = min(15.0, DEADLINE - 15 - (time.time() - start_time))
            if time_budget < 5:
                break

            circles, score = _best_known(W, H, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = W, H

    # Phase 3: Random restarts if time permits
    remaining = DEADLINE - 15 - (time.time() - start_time)
    if remaining > 8:
        n_restarts = max(1, int(remaining / 8))
        for _ in range(n_restarts):
            if time.time() - start_time > DEADLINE - 12:
                break
            circ_rand = _random_initialization(best_W, best_H, rng)
            circ_opt = _local_optimize(circ_rand, best_W, best_H, max_iter=1500)
            circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], best_W - circ_opt[:, 2])
            circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], best_H - circ_opt[:, 2])
            circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(best_W, best_H) / 2)
            circ_opt, valid_rand = _separate_circles(circ_opt, best_W, best_H)
            if valid_rand:
                s = np.sum(circ_opt[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = circ_opt.copy()

    # Phase 4: Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 20:
        polished = _polish(best_result, best_W, best_H, max_iter=12000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Phase 5: Ultra-tight final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        hp = _high_precision_polish(best_result, best_W, best_H, max_iter=10000)
        hp[:, 0] = np.clip(hp[:, 0], hp[:, 2], best_W - hp[:, 2])
        hp[:, 1] = np.clip(hp[:, 1], hp[:, 2], best_H - hp[:, 2])
        hp[:, 2] = np.clip(hp[:, 2], 0.001, min(best_W, best_H) / 2)
        hp, valid_hp = _separate_circles(hp, best_W, best_H, max_iter=2000)
        if valid_hp:
            s = np.sum(hp[:, 2])
            if s > best_score:
                best_score = s
                best_result = hp.copy()

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
