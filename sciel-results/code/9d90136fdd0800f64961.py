"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + Constraint + Greedy Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import least_squares

N = 21


def _lse(vals, temperature=0.01):
    """Log-sum-exp for smooth max."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 0.0
    vmax = np.max(vals)
    if vmax > 1e6:
        return vmax
    log_sum = np.log(np.sum(np.exp((vals - vmax) / temperature)))
    return vmax + temperature * log_sum


def _vectorized_objective(z, penalty_weight=1000.0, lse_temp=0.01):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty (soft min via LSE)
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


def _epigraph_objective(z, penalty_weight=500.0):
    """
    Epigraph formulation: introduce t as the max radius constraint.
    We want to minimize -sum(r) subject to:
      - t >= 0
      - x_i - r_i >= 0 (left wall)
      - w - x_i - r_i >= 0 (right wall)
      - y_i - r_i >= 0 (bottom wall)
      - h - y_i - r_i >= 0 (top wall)
      - (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2 (non-overlap)
    
    The epigraph variable t helps by making the objective smoother.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Non-overlap: use smooth penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Violation: r_sum - d (where d = sqrt(d_sq))
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Wall violations
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
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


def _greedy_expansion(x, y, r, w, h, time_budget):
    """
    Greedy expansion: iteratively grow each circle to its max possible radius,
    then try to shift it to create more space. Repeat for multiple passes.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > time_budget - 2:
            break

        improved = False
        # Process smallest circles first
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget - 2:
                break

            # Try to expand radius i to max possible
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-6:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    if mid > min(x[i], w - x[i], y[i], h - y[i]):
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

            # Try to shift circle i away from its nearest constraint
            min_dist = float('inf')
            move_x, move_y = 0.0, 0.0
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            if d_left < min_dist:
                min_dist = d_left
                move_x, move_y = 1.0, 0.0
            if d_right < min_dist:
                min_dist = d_right
                move_x, move_y = -1.0, 0.0
            if d_bottom < min_dist:
                min_dist = d_bottom
                move_x, move_y = 0.0, 1.0
            if d_top < min_dist:
                min_dist = d_top
                move_x, move_y = 0.0, -1.0

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    if d > 1e-6:
                        move_x, move_y = dx / d, dy / d

            if min_dist < 0.05:
                for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                # Also try perpendicular moves
                perp_x, perp_y = -move_y, move_x
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

    # Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return x, y, r


def _polish_result(z, time_budget):
    """Polish a valid packing with per-circle radius growth and position adjustment."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Force-directed relaxation to resolve any overlaps
    dt = 0.002
    max_iter = int(min(2000, time_budget * 80))
    for it in range(max_iter):
        if time.time() > time_budget - 5:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion
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

    # Phase 2: Per-circle radius growth with position optimization
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
                    # Try to shift toward center
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

    for i in range(n):
        # Left wall: x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        # Right wall: w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        # Bottom wall: y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        # Top wall: h - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

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


def _coordinate_ascent_refine(x, y, r, w, h, time_budget):
    """
    Coordinate ascent: for each circle, optimize its position (x, y) and radius
    while keeping others fixed. This is a 3D sub-problem that can be solved
    more accurately.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    def sub_objective(params, i):
        """Objective for optimizing circle i's (x, y, r) given others fixed."""
        xi = params[0]
        yi = params[1]
        ri = params[2]
        
        # Wall constraints (soft penalty)
        wall_pen = 0.0
        if xi < ri:
            wall_pen += (ri - xi) ** 2
        if xi > w - ri:
            wall_pen += (xi - (w - ri)) ** 2
        if yi < ri:
            wall_pen += (ri - yi) ** 2
        if yi > h - ri:
            wall_pen += (yi - (h - ri)) ** 2
        
        # Overlap with others
        overlap_pen = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = xi - x[j]
            dy = yi - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            overlap = max(0.0, ri + r[j] - d)
            overlap_pen += overlap ** 2
        
        # We want to maximize ri, so minimize -ri + penalties
        return -ri + 1000 * (wall_pen + overlap_pen)

    bounds_i = [(max(0, x[i] - 0.1), min(w, x[i] + 0.1)),
                (max(0, y[i] - 0.1), min(h, y[i] + 0.1)),
                (0.001, min(x[i], w - x[i], y[i], h - y[i]) + 0.05)]

    for i in range(n):
        if time.time() > time_budget - 2:
            break
        
        x0 = [x[i], y[i], r[i]]
        try:
            res = minimize(sub_objective, x0, args=(i,), method='L-BFGS-B',
                           bounds=bounds_i, options={'maxiter': 200, 'ftol': 1e-14})
            if res.success and res.x[2] > r[i] - 1e-6:
                # Validate the new position
                new_x, new_y, new_r = res.x
                if _is_valid_move(x, y, r, w, h, i, new_x, new_y) and new_r <= _max_radius_for_circle(x, y, r, w, h, i) + 1e-6:
                    x[i] = new_x
                    y[i] = new_y
                    r[i] = min(new_r, _max_radius_for_circle(x, y, r, w, h, i))
        except Exception:
            pass

    return x, y, r


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

    # Generate diverse initial guesses
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

        # Random starts
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

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
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

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Polish with per-circle growth
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Greedy expansion
        if best_result is not None and time.time() < deadline - 20:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            
            x_new, y_new, r_new = _greedy_expansion(x, y, r, w, h, deadline - (time.time() - start_time) - 12)
            z_new = np.concatenate([np.column_stack([x_new, y_new, r_new]).flatten(), [w]])
            valid, val = check_validity(z_new)
            if valid and val > best_value:
                best_value = val
                best_result = z_new

        # 3. Coordinate ascent refinement
        if best_result is not None and time.time() < deadline - 10:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            
            x_ca, y_ca, r_ca = _coordinate_ascent_refine(x, y, r, w, h, deadline - (time.time() - start_time) - 5)
            z_ca = np.concatenate([np.column_stack([x_ca, y_ca, r_ca]).flatten(), [w]])
            valid, val = check_validity(z_ca)
            if valid and val > best_value:
                best_value = val
                best_result = z_ca

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 3)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final polish
        if best_result is not None and time.time() < deadline - 3:
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

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
