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

    # For each circle i, the soft-min radius is softmin of:
    #   boundary distances (4 values) and half-distances to all other circles (n-1 values)
    # We build a matrix M[i, k] where k indexes the constraints for circle i
    # To keep it efficient, we compute softmin via log-sum-exp per row.

    # Number of constraints per circle: 4 boundary + (n-1) pairwise
    n_constraints = 4 + (n - 1)
    # M[i, 0:4] = boundary dists, M[i, 4:] = half-distances to others
    M = np.zeros((n, n_constraints))
    M[:, 0] = b_left
    M[:, 1] = b_bot
    M[:, 2] = b_right
    M[:, 3] = b_top

    # Fill pairwise half-distances: for circle i, the constraint from pair (i,j)
    # We need to map triu indices to per-circle constraint slots
    # For circle i, constraints from j < i are at positions 4 + (j) [since j ranges 0..i-1]
    # For circle i, constraints from j > i are at positions 4 + (i) + (j - i - 1)
    # Actually let's just build it directly:
    # For each pair (i,j) with i<j, the half_dist goes into:
    #   M[i, 4 + (j - i - 1)] and M[j, 4 + (i)]
    # Wait, that's not right either. Let me think more carefully.
    # For circle i, the pairwise constraints are with all j != i.
    # Let's index them: for j < i: slot = 4 + j; for j > i: slot = 4 + (j - 1)
    # No, simpler: for circle i, there are n-1 pairwise constraints.
    # Let's just use a flat approach: build the full distance matrix, set diag to inf,
    # then take softmin over each row combined with boundary.

    # Full pairwise half-distance matrix
    D = np.zeros((n, n))
    D[i_idx, j_idx] = half_dist
    D[j_idx, i_idx] = half_dist
    np.fill_diagonal(D, np.inf)

    # Combined constraint matrix: [boundary_dists (n,4), D (n,n-1 after removing diag)]
    # Actually let's just concatenate: C[i] = [b_left_i, b_bot_i, b_right_i, b_top_i, D[i,0],...,D[i,i-1],D[i,i+1],...]
    # This is getting complex. Let me use a simpler formulation.

    # Alternative: use the fact that softmin of values = -logsumexp(-beta * values) / beta
    # For each circle i: r_i_soft = softmin(boundary_i, {D[i,j] for j != i})
    # = -1/beta * log( sum_k exp(-beta * C_ik) )  where C_ik are the constraint values

    # Build full constraint matrix C of shape (n, 4 + n - 1)
    # But n-1 varies... no, it's always n-1 for each row (excluding self).
    # Let's just build C as (n, 4+n) and set self-distance to +inf (exp(-beta*inf)=0)
    C = np.zeros((n, 4 + n))
    C[:, 0] = b_left
    C[:, 1] = b_bot
    C[:, 2] = b_right
    C[:, 3] = b_top
    C[:, 4:] = D  # D has inf on diagonal

    # Softmin: r_i = -1/beta * log(sum_k exp(-beta * C[i,k]))
    neg_beta_C = -beta * C
    # For numerical stability, subtract max
    max_nc = np.max(neg_beta_C, axis=1, keepdims=True)
    # Handle inf: if all are -inf (shouldn't happen), set to 0
    max_nc = np.where(np.isfinite(max_nc), max_nc, 0.0)
    shifted = neg_beta_C - max_nc
    # exp of very negative numbers -> 0, that's fine
    exp_shifted = np.exp(shifted)
    # Set non-finite to 0
    exp_shifted = np.where(np.isfinite(exp_shifted), exp_shifted, 0.0)
    logsumexp = np.log(np.sum(exp_shifted, axis=1)) + max_nc[:, 0]
    r_soft = -logsumexp / beta

    # Ensure non-negative
    r_soft = np.maximum(r_soft, 0.0)
    energy = -np.sum(r_soft)

    # Gradient: d(energy)/d(v) = -dr_soft/d(v)
    # dr_soft/d(C[i,k]) = -exp(-beta*C[i,k]) / (beta * sum_j exp(-beta*C[i,j]))
    # = -exp_shifted[i,k] / (beta * sum_k exp_shifted[i,k])
    # = -weights[i,k] / beta  where weights[i,k] = exp_shifted[i,k] / sum_k exp_shifted[i,k]

    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    sum_exp = np.where(sum_exp > 0, sum_exp, 1.0)
    weights = exp_shifted / sum_exp  # (n, 4+n)
    dC_dr = -weights / beta  # dr_soft[i] / dC[i,k]

    # Now chain rule: dC[i,k]/d(x_i), dC[i,k]/d(y_i)
    # Boundary: C[i,0]=x_i, C[i,1]=y_i, C[i,2]=1-x_i, C[i,3]=1-y_i
    # Pairwise: C[i, 4+j] = 0.5 * sqrt((x_i-x_j)^2 + (y_i-y_j)^2) for j != i
    #           dC[i,4+j]/dx_i = 0.5 * (x_i - x_j) / dist_ij
    #           dC[i,4+j]/dy_i = 0.5 * (y_i - y_j) / dist_ij

    grad = np.zeros(2 * n)
    # x positions: indices 0..n-1
    # y positions: indices n..2n-1

    # Boundary contributions to x:
    # dC[i,0]/dx_i = 1, dC[i,2]/dx_i = -1
    grad[0:n] += dC_dr[:, 0] * 1.0 + dC_dr[:, 2] * (-1.0)
    # Boundary contributions to y:
    grad[n:2*n] += dC_dr[:, 1] * 1.0 + dC_dr[:, 3] * (-1.0)

    # Pairwise contributions:
    # For pair (i,j), i<j:
    #   dC[i,4+j]/dx_i = 0.5*(x_i-x_j)/dist, dC[j,4+i]/dx_j = 0.5*(x_j-x_i)/dist
    #   Similarly for y
    dx_full = xs[:, None] - xs[None, :]
    dy_full = ys[:, None] - ys[None, :]
    dist_full = np.sqrt(dx_full ** 2 + dy_full ** 2 + 1e-30)
    np.fill_diagonal(dist_full, 1.0)  # avoid div by zero (weight will be 0 due to inf)

    # dC[i,4+j]/dx_i = 0.5 * dx_full[i,j] / dist_full[i,j]
    # Contribution to grad[x_i] from all j: sum_j dC_dr[i,4+j] * 0.5 * dx_full[i,j] / dist_full[i,j]
    # But only for j where C[i,4+j] is finite (i.e., j != i)
    # The weight for self (j=i) should be 0 since D[i,i]=inf -> exp(-beta*inf)=0
    # So weights[i, 4+i] should be 0. Let's verify: C[i,4+i] = D[i,i] = inf
    # neg_beta_C[i,4+i] = -beta*inf = -inf, exp(-inf) = 0. Good.

    # Build gradient from pairwise terms
    # grad_x[i] += sum_j dC_dr[i, 4+j] * 0.5 * dx_full[i,j] / dist_full[i,j]
    # grad_y[i] += sum_j dC_dr[i, 4+j] * 0.5 * dy_full[i,j] / dist_full[i,j]

    # Vectorized:
    # W[i,j] = dC_dr[i, 4+j] * 0.5 / dist_full[i,j]
    W = dC_dr[:, 4:] * 0.5 / dist_full  # (n, n)
    # For diagonal, W[i,i] should be 0 (since dC_dr[i,4+i] = 0)
    np.fill_diagonal(W, 0.0)

    grad[0:n] += W @ dx_full  # sum_j W[i,j] * dx_full[i,j]
    grad[n:2*n] += W @ dy_full

    # Wait, let me double-check the sign.
    # energy = -sum(r_soft)
    # d(energy)/d(x_i) = -sum_k d(r_soft_i)/dC[i,k] * dC[i,k]/dx_i
    # = -sum_k dC_dr[i,k] * dC[i,k]/dx_i  (where dC_dr = dr_soft/dC)
    # Hmm, I defined dC_dr = -weights/beta which is dr_soft/dC.
    # So d(energy)/d(x_i) = -sum_k (dr_soft_i/dC[i,k]) * (dC[i,k]/dx_i)
    # = -[boundary part + pairwise part]
    #
    # Let me redo: I computed dC_dr = dr_soft/dC = -weights/beta
    # Then d(energy)/dx_i = -sum_k dC_dr[i,k] * dC[i,k]/dx_i
    #
    # For boundary: dC[i,0]/dx_i=1, dC[i,2]/dx_i=-1
    #   contribution = -(dC_dr[i,0]*1 + dC_dr[i,2]*(-1)) = -(dC_dr[i,0] - dC_dr[i,2])
    #
    # I had: grad[0:n] += dC_dr[:, 0] * 1.0 + dC_dr[:, 2] * (-1.0)
    # That's dC_dr[:,0] - dC_dr[:,2], but I need -(dC_dr[:,0] - dC_dr[:,2])
    # So I need to negate everything!

    grad = -grad

    return energy, grad


def _softmin_polish(pos, deadline, betas=None):
    """
    Polish using smooth soft-min objective with gradient descent.
    """
    if betas is None:
        betas = [20.0, 40.0, 80.0, 160.0, 320.0]

    n = len(pos)
    best_pos = pos.copy()
    r_init = _max_radius_for_positions(best_pos)
    best_obj = float(np.sum(r_init))

    current_v = pos.ravel().copy()

    for beta in betas:
        if time.time() > deadline - 1.0:
            break

        # Run L-BFGS-B on the smooth objective
        v0 = current_v.copy()

        def obj_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return e

        def grad_fn(v, b=beta):
            e, g = _softmin_objective_and_grad(v, n, b)
            return g

        bounds = [(0.0, 1.0)] * (2 * n)

        try:
            res = minimize(
                obj_fn, v0, jac=grad_fn, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
        except Exception:
            continue

        if time.time() > deadline - 0.5:
            break

        v_opt = np.clip(res.x, 0.0, 1.0)
        pos_opt = v_opt.reshape(n, 2)
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
        if _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj:
            best_pos = pos_opt.copy()
            best_obj = obj_opt
            current_v = v_opt.copy()
        elif _is_valid(full_x, n, tol=1e-6):
            current_v = v_opt.copy()

    return best_pos, best_obj


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


def _gradient_polish(pos, deadline, mu=1000.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with soft objective.
    Returns improved (pos, radii) if valid and better.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    def obj_wrapper(v):
        e, g = _soft_objective_and_grad(v, n, mu)
        return e

    def grad_wrapper(v):
        e, g = _soft_objective_and_grad(v, n, mu)
        return g

    try:
        res = minimize(
            obj_wrapper, v0, jac=grad_wrapper, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 80}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]
    r_opt = v_opt[2 * n:3 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    # Build full vector for validity check
    full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
    if _is_valid(full_x, n, tol=1e-6):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: combine soft-min smooth objective and
    progressive-mu gradient optimization for maximum precision.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))

    # Phase A: Soft-min smooth polish (differentiable surrogate)
    if time.time() < deadline - 5:
        sm_pos, sm_obj = _softmin_polish(current_pos, deadline - 4)
        if sm_obj > best_obj:
            best_pos = sm_pos.copy()
            best_obj = sm_obj
            best_r = _max_radius_for_positions(best_pos)
            current_pos = sm_pos.copy()

    # Phase B: Progressive-mu gradient polish
    mus = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 15000.0]

    for mu in mus:
        if time.time() > deadline - 2:
            break

        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        def obj_wrapper(v, m=mu):
            e, g = _soft_objective_and_grad(v, n, m)
            return e

        def grad_wrapper(v, m=mu):
            e, g = _soft_objective_and_grad(v, n, m)
            return g

        try:
            maxiter = 1000 if mu >= 4000 else 600
            res = minimize(
                obj_wrapper, v0, jac=grad_wrapper, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 80}
            )
        except Exception:
            continue

        if time.time() > deadline - 2:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])

        pos_opt = np.clip(pos_opt, 0.0, 1.0)

        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        full_x = np.concatenate([pos_opt[:, 0], pos_opt[:, 1], r_exact])
        if _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(full_x, n, tol=1e-6) and obj_opt > best_obj - 1e-10:
            current_pos = pos_opt.copy()

    if time.time() < deadline - 0.5:
        # Grow radii step
        full_x = np.concatenate([best_pos[:, 0], best_pos[:, 1], best_r])
        grown_x = _grow_radii(full_x, n, max_iter=100, step=0.0005, tol=1e-8)
        grown_pts = grown_x.reshape(n, 3)
        grown_obj = float(np.sum(grown_pts[:, 2]))
        if _is_valid(grown_x, n, tol=1e-6) and grown_obj > best_obj:
            best_pos = grown_pts[:, :2]
            best_r = grown_pts[:, 2]
            best_obj = grown_obj

        # Final fine-tuning
        rng = np.random.RandomState(42)
        for _ in range(800):
            if time.time() > deadline - 0.1:
                break
            i = rng.randint(n)
            dx = rng.uniform(-0.0004, 0.0004)
            dy = rng.uniform(-0.0004, 0.0004)
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

    # Diverse initializations (reduced to 4 to save time for polish)
    initializations = [_initial_grid()]
    hx, hy = _hex_packing_init(N)
    initializations.append(np.column_stack([hx, hy]))
    for s in range(2):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 50:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=15000, seed=init_idx * 7 + 1,
            deadline=deadline - 45
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 35:
            break

        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=4000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 30
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        if time.time() > deadline - 25:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=1500, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 22
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 30:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=8000,
            seed=777, deadline=deadline - 25
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 2b: Second best-fit pass
    if time.time() < deadline - 22:
        bf_final2, bf_final2_obj = _best_fit_local_search(
            best_pos, n_iter=5000,
            seed=888, deadline=deadline - 18
        )
        if bf_final2_obj > best_obj:
            best_pos = bf_final2
            best_obj = bf_final2_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 15:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=3000, step_size=0.002,
            seed=999, deadline=deadline - 12
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Bubble relaxation
    if time.time() < deadline - 22:
        bubble_pos, bubble_obj = _bubble_relaxation(
            best_pos, deadline=deadline - 18,
            n_iter=2000, dt=0.001
        )
        if bubble_obj > best_obj:
            best_pos = bubble_pos
            best_obj = bubble_obj

    # Phase 5: Gradient-based polish
    if time.time() < deadline - 20:
        polished = _gradient_polish(best_pos, deadline - 17, mu=1000.0)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 6: High-precision final polish (soft-min + progressive mu)
    if time.time() < deadline - 12:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 5)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 7: Another best-fit pass after gradient polish
    if time.time() < deadline - 6:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=555, deadline=deadline - 4
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 8: Final fine-tuning local search
    if time.time() < deadline - 3:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1000, step_size=0.0006,
            seed=12345, deadline=deadline - 2
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 9: Ultra-fine polish
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(400):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0002, 0.0002)
            dy = rng.uniform(-0.0002, 0.0002)
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
