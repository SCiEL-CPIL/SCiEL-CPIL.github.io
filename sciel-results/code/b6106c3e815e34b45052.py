"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Deadline guard: 140s wall-clock (10s safety margin under 150s)
_DEADLINE = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    from scipy.spatial.distance import cdist
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _soft_objective_and_grad(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _is_valid(x, n=N, tol=1e-6):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _project_feasible(x, n=N, max_iter=500, tol=1e-10):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    Reduces radii and adjusts positions to eliminate boundary/overlap violations.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

        # Fix boundary violations: reduce radius to fit
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-9)
            pts[:, 2] = rs

        # Fix overlap violations: reduce radii
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)

        if not np.any(ov > tol):
            break

        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)

        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.01, rs - reduction)
        pts[:, 2] = rs

    return pts.ravel()


def _grow_radii(x, n=N, max_iter=200, step=0.001, tol=1e-8):
    """
    Try to grow radii while maintaining feasibility.
    Uses a greedy approach: try to increase each radius slightly,
    check feasibility, and keep the increase if valid.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        if _remaining() < 2.0:
            break

        improved = False
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

        for i in range(n):
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = 1.0 - ys[i]
            max_r_boundary = min(d_left, d_bot, d_right, d_top)

            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                limit = dist - rs[j]
                if limit < max_r_overlap:
                    max_r_overlap = limit

            new_r = min(max_r_overlap, rs[i] + step)
            if new_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = new_r
                pts[i, 2] = new_r

                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r

        if not improved:
            break

    return pts.ravel()


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est

    xs, ys = [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        x_start = (1.0 - total_width) / 2.0
        for j in range(count):
            xs.append(x_start + j * x_spacing)
            ys.append(y)

    while len(xs) < n:
        xs.append(0.5)
        ys.append(0.5)

    return np.array(xs[:n]), np.array(ys[:n])


def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """
    Simulated annealing to maximize sum of radii.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        t = t_start * (t_end / t_start) ** (it / n_iter)

        i = rng.randint(n)
        step_size = 0.04 * t + 0.0008
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)

        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

        new_obj = _objective(new_pos)
        delta = new_obj - current_obj

        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj

    return best_pos, best_obj


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """
    Local search: iteratively nudge each circle to improve the objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(15):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break

    return current_pos, current_obj


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    base_step = 0.015
    grid_res = 5

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)

        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]

            best_new_pos = None
            best_new_obj = current_obj

            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_pos = new_pos

            for _ in range(8):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_pos = new_pos

            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                x_i, y_i = current_pos[i]
                dx = rng.uniform(-step * 0.5, step * 0.5)
                dy = rng.uniform(-step * 0.5, step * 0.5)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(x_i + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(y_i + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-14:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                base_step *= 0.6
                if base_step < 0.0003:
                    break

    return current_pos, current_obj


def _bubble_relaxation(pos, deadline, n_iter=2000, dt=0.001):
    """
    Physics-based bubble relaxation.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_obj = float(np.sum(r_current))

    vx = np.zeros(n)
    vy = np.zeros(n)

    damping = 0.95

    for it in range(n_iter):
        if time.time() > deadline:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            x, y = current_pos[i]
            r = r_current[i]

            if x < r:
                fx[i] += (r - x) * 100.0
            if x > 1.0 - r:
                fx[i] -= (x - (1.0 - r)) * 100.0
            if y < r:
                fy[i] += (r - y) * 100.0
            if y > 1.0 - r:
                fy[i] -= (y - (1.0 - r)) * 100.0

        for i in range(n):
            for j in range(i + 1, n):
                dx = current_pos[j, 0] - current_pos[i, 0]
                dy = current_pos[j, 1] - current_pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < 1e-10:
                    dx = np.random.uniform(-0.001, 0.001)
                    dy = np.random.uniform(-0.001, 0.001)
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-10:
                        continue

                overlap = r_current[i] + r_current[j] - dist

                if overlap > 0:
                    force_mag = overlap * 50.0
                    fx[i] -= force_mag * dx / dist
                    fy[i] -= force_mag * dy / dist
                    fx[j] += force_mag * dx / dist
                    fy[j] += force_mag * dy / dist

        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt

        current_pos[:, 0] += vx * dt
        current_pos[:, 1] += vy * dt

        current_pos[:, 0] = np.clip(current_pos[:, 0], 0.0, 1.0)
        current_pos[:, 1] = np.clip(current_pos[:, 1], 0.0, 1.0)

        r_new = _max_radius_for_positions(current_pos)
        obj_new = float(np.sum(r_new))

        if obj_new > best_obj + 1e-12:
            best_pos = current_pos.copy()
            best_obj = obj_new
            r_current = r_new

    return best_pos, best_obj


def _softmin_objective_and_grad(v, n=N, beta=50.0):
    """
    Smooth surrogate: maximize sum of soft-min radii.
    v = [x0,y0,...,xn-1,yn-1] (2n values, positions only).
    Uses log-sum-exp for smooth min of pairwise distances and boundary distances.
    Returns (energy, grad) where energy = -sum(softmin_radii).
    """
    pos = v.reshape(n, 2)
    xs, ys = pos[:, 0], pos[:, 1]

    # Boundary distances for each circle
    b_left = xs
    b_bot = ys
    b_right = 1.0 - xs
    b_top = 1.0 - ys

    # Pairwise half-distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq + 1e-30)
    half_dist = dist / 2.0

    # For each circle i, soft-min radius = softmin of boundary dists and half-distances to others
    # We compute this per-circle
    # Collect all candidate distances for each circle
    # Use a matrix approach: for circle i, candidates are [b_left_i, b_bot_i, b_right_i, b_top_i, half_dist(i,j) for j!=i]

    # Build a (n, n+3) matrix of candidate distances (padding with large values)
    # Actually, let's do it more directly per circle using the fact that n is small (26)

    softmin_radii = np.zeros(n)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)

    for i in range(n):
        # Candidates: 4 boundary + (n-1) half-distances
        cands = np.array([b_left[i], b_bot[i], b_right[i], b_top[i]])
        for j in range(n):
            if j != i:
                cands = np.append(cands, half_dist[np.where((i_idx == i) | (j_idx == i))[0][0]] if False else None)

        # Let me redo this more carefully
        cands = np.zeros(4 + n - 1)
        cands[0] = b_left[i]
        cands[1] = b_bot[i]
        cands[2] = b_right[i]
        cands[3] = b_top[i]
        for k, j in enumerate(range(n)):
            if j == i:
                continue
            # find the index in i_idx/j_idx for pair (i,j)
            # This is inefficient but n=26 so it's fine
            # Better: precompute distance matrix
            pass

    # Let me use a cleaner approach with full distance matrix
    from scipy.spatial.distance import cdist
    D = cdist(pos, pos)
    np.fill_diagonal(D, np.inf)
    half_D = D / 2.0  # (n, n), inf on diagonal

    # For each circle i: candidates are [b_left_i, b_bot_i, b_right_i, b_top_i, half_D[i,0], ..., half_D[i,n-1]]
    # (with half_D[i,i] = inf, which won't affect softmin)
    # Stack: (n, n+4)
    boundary = np.column_stack([b_left, b_bot, b_right, b_top])  # (n, 4)
    all_cands = np.column_stack([boundary, half_D])  # (n, n+4)

    # Softmin via log-sum-exp: softmin(a) = -logsumexp(-beta * a) / beta
    # For numerical stability
    neg_beta_cands = -beta * all_cands
    # Clip inf to large negative for logsumexp
    neg_beta_cands = np.where(np.isinf(neg_beta_cands), -1e10, neg_beta_cands)

    max_val = np.max(neg_beta_cands, axis=1, keepdims=True)
    shifted = neg_beta_cands - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True)) + max_val
    softmin_radii = -log_sum_exp / beta  # (n, 1) -> (n,)
    softmin_radii = softmin_radii.flatten()

    energy = -np.sum(softmin_radii)

    # Gradient: d(softmin_i)/d(x_i) etc.
    # softmin_i = -logsumexp(-beta * cands_i) / beta
    # d(softmin_i)/d(cand_k) = (beta * exp(-beta * cand_k)) / (sum_j exp(-beta * cand_j))
    #                        = exp(shifted_k) / sum(exp(shifted_j))  (probability)
    # But with the sign: d(softmin)/d(cand) = -exp(shifted_k)/sum(exp(shifted))
    # Wait: softmin = -(1/beta)*logsumexp(-beta*c)
    # d(softmin)/d(c_k) = -(1/beta) * (-beta * exp(-beta*c_k)) / sum(exp(-beta*c_j))
    #                    = exp(-beta*c_k) / sum(exp(-beta*c_j))
    #                    = exp(shifted_k) / sum(exp(shifted))

    exp_shifted = np.exp(shifted)  # (n, n+4)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)  # (n, 1)
    weights = exp_shifted / sum_exp  # (n, n+4)

    # d(energy)/d(x) = -sum_i d(softmin_i)/d(x)
    # d(softmin_i)/d(x_i) = weights[i, 0] * d(b_left_i)/d(x_i) + weights[i, 2] * d(b_right_i)/d(x_i)
    #   = weights[i, 0] * 1 + weights[i, 2] * (-1)
    # d(softmin_i)/d(x_j) for j != i: from half_D[i,j] = dist(i,j)/2
    #   d(dist(i,j))/d(x_i) = (x_i - x_j) / dist(i,j)
    #   d(half_D[i,j])/d(x_i) = (x_i - x_j) / (2 * dist(i,j))
    #   contribution to d(softmin_i)/d(x_i) from pair (i,j): weights[i, j+4] * (x_i-x_j)/(2*dist)
    #   contribution to d(softmin_j)/d(x_i) from pair (i,j): weights[j, i+4] * (x_i-x_j)/(2*dist)

    # Let's compute gradient numerically-safe:
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)

    for i in range(n):
        # Boundary contributions
        grad_x[i] += weights[i, 0] * 1.0   # b_left = x_i, d/dx_i = 1
        grad_x[i] += weights[i, 2] * (-1.0)  # b_right = 1-x_i, d/dx_i = -1
        grad_y[i] += weights[i, 1] * 1.0   # b_bot = y_i
        grad_y[i] += weights[i, 3] * (-1.0)  # b_top = 1-y_i

        # Pairwise contributions
        for j in range(n):
            if j == i:
                continue
            # weight for half_dist(i,j) in softmin_i is weights[i, j+4]
            # weight for half_dist(j,i) in softmin_j is weights[j, i+4]
            dx_ij = xs[i] - xs[j]
            dy_ij = ys[i] - ys[j]
            d_ij = np.sqrt(dx_ij ** 2 + dy_ij ** 2 + 1e-30)
            # d(half_D[i,j])/d(x_i) = dx_ij / (2 * d_ij)
            # d(half_D[i,j])/d(x_j) = -dx_ij / (2 * d_ij)
            # d(softmin_i)/d(x_i) += weights[i, j+4] * dx_ij / (2*d_ij)
            # d(softmin_i)/d(x_j) += weights[i, j+4] * (-dx_ij / (2*d_ij))
            # d(softmin_j)/d(x_i) += weights[j, i+4] * dx_ij / (2*d_ij)
            # d(softmin_j)/d(x_j) += weights[j, i+4] * (-dx_ij / (2*d_ij))

            w_i = weights[i, j + 4]
            w_j = weights[j, i + 4]

            # d(energy)/d(x_i) = -d(softmin_i)/d(x_i) - d(softmin_j)/d(x_i)
            # = -w_i * dx_ij/(2*d_ij) - w_j * dx_ij/(2*d_ij)
            grad_x[i] -= (w_i + w_j) * dx_ij / (2.0 * d_ij)
            grad_x[j] -= (w_i + w_j) * (-dx_ij) / (2.0 * d_ij)

            grad_y[i] -= (w_i + w_j) * dy_ij / (2.0 * d_ij)
            grad_y[j] -= (w_i + w_j) * (-dy_ij) / (2.0 * d_ij)

    grad = np.concatenate([grad_x, grad_y])
    return energy, grad


def _softmin_polish(pos, deadline, betas=None):
    """
    Polish positions using soft-min surrogate with iterative beta sharpening.
    """
    if betas is None:
        betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 8000.0]

    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))

    for beta in betas:
        if time.time() > deadline - 1.0:
            break

        v0 = current_pos.ravel().copy()

        def obj_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return e

        def grad_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return g

        bounds = [(0.0, 1.0)] * (2 * n)

        try:
            maxiter = 800 if beta >= 1000 else 500
            res = minimize(
                obj_fn, v0, jac=grad_fn, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
        except Exception:
            continue

        if time.time() > deadline - 1.0:
            break

        pos_opt = res.x.reshape(n, 2)
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
        if _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj:
            best_pos = pos_opt.copy()
            best_r = r_exact.copy()
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(full_x, n, tol=1e-6):
            current_pos = pos_opt.copy()

    return best_pos, best_r, best_obj


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: use soft-min surrogate with progressive beta sharpening,
    then grow radii and fine-tune.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    best_pos = pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))

    # Phase A: Soft-min polish with progressive beta
    if time.time() < deadline - 3:
        sm_pos, sm_r, sm_obj = _softmin_polish(
            pos, deadline=deadline - 3,
            betas=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 8000.0]
        )
        if sm_obj > best_obj:
            best_pos = sm_pos
            best_r = sm_r
            best_obj = sm_obj

    # Phase B: Grow radii
    if time.time() < deadline - 1.5:
        full_x = np.concatenate([best_pos[:, 0], best_pos[:, 1], best_r])
        grown_x = _grow_radii(full_x, n, max_iter=150, step=0.0005, tol=1e-8)
        grown_pts = grown_x.reshape(n, 3)
        grown_obj = float(np.sum(grown_pts[:, 2]))
        if _is_valid(grown_x, n, tol=1e-6) and grown_obj > best_obj:
            best_pos = grown_pts[:, :2]
            best_r = grown_pts[:, 2]
            best_obj = grown_obj

    # Phase C: Another soft-min pass at higher betas on the grown config
    if time.time() < deadline - 1.0:
        sm2_pos, sm2_r, sm2_obj = _softmin_polish(
            best_pos, deadline=deadline - 0.5,
            betas=[500.0, 2000.0, 5000.0, 10000.0]
        )
        if sm2_obj > best_obj:
            best_pos = sm2_pos
            best_r = sm2_r
            best_obj = sm2_obj

    # Phase D: Final fine-tuning
    if time.time() < deadline - 0.1:
        rng = np.random.RandomState(42)
        for _ in range(600):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(n)
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            full_x = np.concatenate([new_pos[:, 0], new_pos[:, 1], new_r])
            if _is_valid(full_x, n, tol=1e-6) and new_obj > best_obj + 1e-12:
                best_pos = new_pos
                best_r = new_r
                best_obj = new_obj

    return best_pos, best_r, best_obj


def construct():
    global _START
    _START = time.time()
    start_time = _START
    deadline = start_time + _DEADLINE

    # Diverse initializations
    initializations = [_initial_grid()]
    hx, hy = _hex_packing_init(N)
    initializations.append(np.column_stack([hx, hy]))
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 40:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=25000, seed=init_idx * 7 + 1,
            deadline=deadline - 35
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 25:
            break

        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=5000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 20
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        if time.time() > deadline - 15:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 12
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 25:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=10000,
            seed=777, deadline=deadline - 20
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 2b: Second best-fit pass
    if time.time() < deadline - 18:
        bf_final2, bf_final2_obj = _best_fit_local_search(
            best_pos, n_iter=7000,
            seed=888, deadline=deadline - 15
        )
        if bf_final2_obj > best_obj:
            best_pos = bf_final2
            best_obj = bf_final2_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 12:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=4000, step_size=0.002,
            seed=999, deadline=deadline - 10
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Bubble relaxation
    if time.time() < deadline - 20:
        bubble_pos, bubble_obj = _bubble_relaxation(
            best_pos, deadline=deadline - 15,
            n_iter=2500, dt=0.001
        )
        if bubble_obj > best_obj:
            best_pos = bubble_pos
            best_obj = bubble_obj

    # Phase 5: Soft-min gradient polish (progressive beta)
    if time.time() < deadline - 20:
        sm_pos, sm_r, sm_obj = _softmin_polish(
            best_pos, deadline=deadline - 15,
            betas=[10.0, 50.0, 200.0, 1000.0, 5000.0]
        )
        if sm_obj > best_obj:
            best_pos = sm_pos
            best_obj = sm_obj

    # Phase 6: High-precision final polish (soft-min + grow + fine-tune)
    if time.time() < deadline - 8:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 3)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 7: Another best-fit pass after gradient polish
    if time.time() < deadline - 5:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 8: Final fine-tuning local search
    if time.time() < deadline - 2:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1000, step_size=0.0004,
            seed=12345, deadline=deadline - 1
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 9: Ultra-fine polish
    if time.time() < deadline - 0.1:
        rng = np.random.RandomState(99999)
        for _ in range(300):
            if time.time() > deadline - 0.02:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.00015, 0.00015)
            dy = rng.uniform(-0.00015, 0.00015)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if new_obj > best_obj + 1e-13:
                best_pos = new_pos
                best_obj = new_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
