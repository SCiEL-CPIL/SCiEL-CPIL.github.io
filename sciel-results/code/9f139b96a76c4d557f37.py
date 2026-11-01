"""Circle packing in perimeter-4 rectangle: SLSQP + PBD + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize, least_squares, differential_evolution

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _pbd_optimize(x, y, r, w, h, time_budget, n_iters=5000):
    """
    Determinant-based area optimization: reformulate as finding the minimal
    bounding rectangle for fixed radii, then determine max uniform scale.
    Uses SLSQP on positions with a smooth objective that penalizes both
    overlap and excessive bounding box, enabling a different search landscape.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N

    def _area_obj(params, r_scale):
        """Objective: minimize sum of radii * scale (to maximize scale) + penalties."""
        vx = params[0::2][:n]
        vy = params[1::2][:n]
        rs = r * r_scale

        if np.any(rs < 0) or np.any(vx < 0) or np.any(vy < 0):
            return 1e6

        # Bounding box
        xmin = np.min(vx - rs)
        xmax = np.max(vx + rs)
        ymin = np.min(vy - rs)
        ymax = np.max(vy + rs)
        bw = xmax - xmin
        bh = ymax - ymin

        # Perimeter constraint: bw + bh <= 4 (we want to maximize scale, so minimize perimeter usage)
        perim = bw + bh
        if perim > 4.0 + 1e-6:
            return 1e6

        # Objective: minimize negative scale (i.e., maximize scale)
        # Plus overlap penalties
        obj = -r_scale

        # Overlap penalty
        dx = vx[:, None] - vx[None, :]
        dy = vy[:, None] - vy[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rs[:, None] + rs[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        obj += 500.0 * np.sum(overlap * overlap)

        # Keep bounding box within perimeter 4
        if perim > 4.0:
            obj += 1000.0 * (perim - 4.0) ** 2

        return obj

    def _find_max_scale(x_init, y_init, r_base, time_limit):
        """Binary search for max uniform scale factor."""
        lo, hi = 1.0, 2.0
        # First check if scale=1 is feasible
        if not _is_valid(x_init, y_init, r_base * 1.0, w, h):
            return x_init, y_init, r_base

        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_base.copy()
        best_scale = 1.0

        for _ in range(25):
            if time.time() > time_limit:
                break
            mid = (lo + hi) / 2.0
            r_trial = r_base * mid

            # Try to optimize positions for this scale
            params0 = np.empty(2 * n)
            params0[0::2] = x_init
            params0[1::2] = y_init

            bounds_pos = [(0.0, 2.0)] * (2 * n)

            try:
                res = minimize(
                    lambda p: _area_obj(p, mid),
                    params0,
                    method='SLSQP',
                    bounds=bounds_pos,
                    options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50}
                )
                x_opt = res.x[0::2][:n]
                y_opt = res.x[1::2][:n]

                if _is_valid(x_opt, y_opt, r_trial, w, h):
                    best_x, best_y, best_r = x_opt, y_opt, r_trial
                    best_scale = mid
                    lo = mid
                else:
                    hi = mid
            except Exception:
                hi = mid

        return best_x, best_y, best_r

    # Phase 1: Optimize positions for current scale
    params0 = np.empty(2 * n)
    params0[0::2] = x
    params0[1::2] = y

    bounds_pos = [(0.0, 2.0)] * (2 * n)

    try:
        res = minimize(
            lambda p: _area_obj(p, 1.0),
            params0,
            method='SLSQP',
            bounds=bounds_pos,
            options={'maxiter': 400, 'ftol': 1e-12, 'maxls': 60}
        )
        x_opt = res.x[0::2][:n]
        y_opt = res.x[1::2][:n]
        if _is_valid(x_opt, y_opt, r, w, h):
            x, y = x_opt, y_opt
    except Exception:
        pass

    # Phase 2: Binary search for max uniform scale
    x_new, y_new, r_new = _find_max_scale(x, y, r, time_budget - 3)

    if _is_valid(x_new, y_new, r_new, w, h):
        return x_new, y_new, r_new, w, h
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _polish_result(z, time_budget):
    """Polish a valid packing with per-circle radius growth and position adjustment."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Force-directed relaxation
    dt = 0.002
    max_iter = int(min(2000, time_budget * 80))
    for it in range(max_iter):
        if time.time() > time_budget - 5:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-10:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-9:
            break

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.001
        else:
            dt = 0.002

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Phase 2: Per-circle radius growth
    order = np.argsort(r)
    for _pass in range(3):
        if time.time() > time_budget - 3:
            break
        for i in order:
            if time.time() > time_budget - 3:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    cx = (x[i] - r[i] + w - x[i] - r[i]) / 2 + r[i]
                    cy = (y[i] - r[i] + h - y[i] - r[i]) / 2 + r[i]
                    for step in [0.01, 0.005, 0.002]:
                        nx = x[i] + (cx - x[i]) * step
                        ny = y[i] + (cy - y[i]) * step
                        if _is_valid_move(x, y, r, w, h, i, nx, ny):
                            x[i] = nx
                            y[i] = ny
                            break

    # Phase 3: Uniform growth
    grow_factor = 1.0005
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget - 2:
            break
        improved = False
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget - 2:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    if mid > r_wall:
                        hi = mid
                        continue
                    ok = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d2 = dx * dx + dy * dy
                        rs = mid + r[j]
                        if d2 < rs * rs - 1e-9:
                            ok = False
                            break
                    if ok:
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True

            if time.time() > time_budget - 2:
                break
            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            if min_dist < 0.03:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.02, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    grow_factor = 1.001
    for _ in range(150):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _epigraph_refine(z, time_budget):
    """
    Smooth epigraph refinement using log-sum-exp to handle non-convex constraints.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    beta = 50.0
    lambda_t = 100.0

    def epigraph_objective(params):
        vx = params[0::3][:n]
        vy = params[1::3][:n]
        vr = params[2::3][:n]
        vw = params[-2]
        vt = params[-1]
        vh = 2.0 - vw

        if vw < 0.1 or vh < 0.1 or np.any(vr < 0):
            return 1e6

        violations = []
        left_v = np.maximum(vr - vx, 0.0)
        right_v = np.maximum(vx + vr - vw, 0.0)
        bottom_v = np.maximum(vr - vy, 0.0)
        top_v = np.maximum(vy + vr - vh, 0.0)
        violations.extend(left_v.tolist())
        violations.extend(right_v.tolist())
        violations.extend(bottom_v.tolist())
        violations.extend(top_v.tolist())

        for i in range(n):
            for j in range(i + 1, n):
                dx = vx[i] - vx[j]
                dy = vy[i] - vy[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = vr[i] + vr[j] - d
                violations.append(max(overlap, 0.0))

        violations = np.array(violations)
        max_viol = np.max(violations) if len(violations) > 0 else 0.0

        if max_viol < 1e-10:
            lse = 0.0
        else:
            shifted = violations - max_viol
            lse = max_viol + np.log(np.sum(np.exp(shifted))) / beta

        obj = -np.sum(vr) + lambda_t * lse
        if vt < lse - 1e-6:
            obj += lambda_t * (lse - vt) ** 2
        return obj

    params0 = np.zeros(3 * n + 2)
    params0[0::3][:n] = x
    params0[1::3][:n] = y
    params0[2::3][:n] = r
    params0[-2] = w
    params0[-1] = 0.0

    bounds_ep = []
    for i in range(n):
        bounds_ep.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds_ep.extend([(0.3, 1.7), (0, 1.0)])

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(epigraph_objective, params0, method='SLSQP',
                       bounds=bounds_ep,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        params_opt = res.x
        x_opt = params_opt[0::3][:n]
        y_opt = params_opt[1::3][:n]
        r_opt = params_opt[2::3][:n]
        w_opt = params_opt[-2]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x_opt
            z_new[1::3][:n] = y_opt
            z_new[2::3][:n] = r_opt
            z_new[-1] = w_opt
            return z_new
        return None
    except Exception:
        return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _contact_graph_optimize(z, time_budget):
    """
    Identify contact structure from current packing, then solve exact tangency
    equations to find a more optimal configuration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    contact_tol = 0.015
    contacts = []
    wall_contacts = []

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[i] + r[j]
            if abs(d - r_sum) < contact_tol * max(r_sum, 0.1):
                contacts.append((i, j))

    wall_tol = 0.01
    for i in range(n):
        if abs(x[i] - r[i]) < wall_tol:
            wall_contacts.append((i, 0))
        if abs(x[i] + r[i] - w) < wall_tol:
            wall_contacts.append((i, 1))
        if abs(y[i] - r[i]) < wall_tol:
            wall_contacts.append((i, 2))
        if abs(y[i] + r[i] - h) < wall_tol:
            wall_contacts.append((i, 3))

    if len(contacts) < n - 2:
        return None

    n_vars = 3 * n + 1

    def contact_residuals(v):
        vx = v[0::3][:n]
        vy = v[1::3][:n]
        vr = v[2::3][:n]
        vw = v[-1]
        vh = 2.0 - vw
        residuals = []
        for (i, j) in contacts:
            dx = vx[i] - vx[j]
            dy = vy[i] - vy[j]
            d = np.sqrt(dx * dx + dy * dy)
            residuals.append(d - (vr[i] + vr[j]))
        for (i, wall) in wall_contacts:
            if wall == 0:
                residuals.append(vx[i] - vr[i])
            elif wall == 1:
                residuals.append(vx[i] + vr[i] - vw)
            elif wall == 2:
                residuals.append(vy[i] - vr[i])
            elif wall == 3:
                residuals.append(vy[i] + vr[i] - vh)
        return np.array(residuals)

    v0 = np.zeros(n_vars)
    v0[0::3][:n] = x
    v0[1::3][:n] = y
    v0[2::3][:n] = r
    v0[-1] = w

    try:
        result = least_squares(
            contact_residuals, v0,
            bounds=(np.zeros(n_vars), np.ones(n_vars) * 2.0),
            method='trf',
            xtol=1e-14, ftol=1e-14, gtol=1e-14,
            max_nfev=500
        )
        v_sol = result.x
        x_sol = v_sol[0::3][:n].copy()
        y_sol = v_sol[1::3][:n].copy()
        r_sol = v_sol[2::3][:n].copy()
        w_sol = v_sol[-1]
        h_sol = 2.0 - w_sol

        if _is_valid(x_sol, y_sol, r_sol, w_sol, h_sol):
            val = float(np.sum(r_sol))
            current_val = float(np.sum(r))
            if val > current_val * (1 - 1e-6):
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:n] = x_sol
                z_new[1::3][:n] = y_sol
                z_new[2::3][:n] = r_sol
                z_new[-1] = w_sol
                return z_new
    except Exception:
        pass
    return None


def _global_search(z, time_budget):
    """
    Global search using differential evolution on a reduced parameter space.
    We search over aspect ratio and a small number of position perturbations
    to escape local minima.
    """
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]

    best_val = float(np.sum(r0))
    best_z = z.copy()

    # Differential evolution over aspect ratio and global shifts
    def de_objective(params):
        w = params[0]
        h = 2.0 - w
        dx = params[1]
        dy = params[2]
        scale = params[3]

        if w < 0.4 or h < 0.4 or scale < 0.5 or scale > 1.5:
            return 1e6

        x = x0 * scale + dx
        y = y0 * scale + dy
        r = r0 * scale

        # Quick validity check
        if np.any(x - r < 0) or np.any(x + r > w) or np.any(y - r < 0) or np.any(y + r > h):
            return 1e6

        for i in range(n):
            for j in range(i + 1, n):
                ddx = x[i] - x[j]
                ddy = y[i] - y[j]
                if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2:
                    return 1e6

        return -np.sum(r)

    bounds = [(0.5, 1.5), (-0.2, 0.2), (-0.2, 0.2), (0.7, 1.3)]
    try:
        res = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=50, popsize=10,
            tol=1e-6, polish=True
        )
        if res.fun < -best_val + 1e-6:
            w = res.x[0]
            h = 2.0 - w
            dx = res.x[1]
            dy = res.x[2]
            scale = res.x[3]
            x = x0 * scale + dx
            y = y0 * scale + dy
            r = r0 * scale
            if _is_valid(x, y, r, w, h):
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:n] = x
                z_new[1::3][:n] = y
                z_new[2::3][:n] = r
                z_new[-1] = w
                return z_new
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # PBD Optimization (area-based reformulation)
        if best_result is not None and time.time() < deadline - 20:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w

            pbd_result = _pbd_optimize(x, y, r, w, h, deadline - (time.time() - start_time) - 10, n_iters=3000)
            if pbd_result is not None:
                x_r, y_r, r_r, w_r, h_r = pbd_result
                z_pbd = np.empty(3 * N + 1)
                z_pbd[0::3][:N] = x_r
                z_pbd[1::3][:N] = y_r
                z_pbd[2::3][:N] = r_r
                z_pbd[-1] = w_r
                valid, val = check_validity(z_pbd)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_pbd

        # Second pass: try different aspect ratios with the area optimizer
        if best_result is not None and time.time() < deadline - 12:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w

            # Try a few alternative aspect ratios
            for w_try in [w * 0.9, w * 1.1, 1.0, 0.8, 1.2]:
                if time.time() > deadline - 12:
                    break
                h_try = 2.0 - w_try
                if h_try < 0.3 or w_try < 0.3:
                    continue
                # Rescale positions to fit new rectangle
                x_try = x * (w_try / w)
                y_try = y * (h_try / h)
                r_try = r * min(w_try / w, h_try / h) * 0.95

                if not _is_valid(x_try, y_try, r_try, w_try, h_try):
                    continue

                pbd_result2 = _pbd_optimize(x_try, y_try, r_try, w_try, h_try,
                                            deadline - (time.time() - start_time) - 8, n_iters=2000)
                if pbd_result2 is not None:
                    x_r, y_r, r_r, w_r, h_r = pbd_result2
                    z_pbd2 = np.empty(3 * N + 1)
                    z_pbd2[0::3][:N] = x_r
                    z_pbd2[1::3][:N] = y_r
                    z_pbd2[2::3][:N] = r_r
                    z_pbd2[-1] = w_r
                    valid, val = check_validity(z_pbd2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_pbd2

        # Contact graph optimization
        if best_result is not None and time.time() < deadline - 10:
            cg_result = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 5)
            if cg_result is not None:
                valid, val = check_validity(cg_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result

        # Post-contact-graph polish with area optimizer
        if best_result is not None and time.time() < deadline - 8:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            pbd_result3 = _pbd_optimize(x, y, r, w, h, deadline - (time.time() - start_time) - 4, n_iters=2000)
            if pbd_result3 is not None:
                x_r, y_r, r_r, w_r, h_r = pbd_result3
                z_pbd3 = np.empty(3 * N + 1)
                z_pbd3[0::3][:N] = x_r
                z_pbd3[1::3][:N] = y_r
                z_pbd3[2::3][:N] = r_r
                z_pbd3[-1] = w_r
                valid, val = check_validity(z_pbd3)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_pbd3

        # Global search
        if best_result is not None and time.time() < deadline - 10:
            gs_result = _global_search(best_result, deadline - (time.time() - start_time) - 5)
            if gs_result is not None:
                valid, val = check_validity(gs_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = gs_result

        # Second polish pass
        if best_result is not None and time.time() < deadline - 10:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 5)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        # Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 5:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Epigraph refinement
        if best_result is not None and time.time() < deadline - 5:
            epi_refined = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 2)
            if epi_refined is not None:
                valid, val = check_validity(epi_refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi_refined
                    if time.time() < deadline - 3:
                        refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 1)
                        if refined2 is not None:
                            valid, val = check_validity(refined2)
                            if valid and val > best_value:
                                best_value = val
                                best_result = refined2

        # Final polish
        if best_result is not None and time.time() < deadline - 2:
            polished3 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished3 is not None:
                valid, val = check_validity(polished3)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished3

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
