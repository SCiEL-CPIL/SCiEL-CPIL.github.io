"""Circle packing in perimeter-4 rectangle: vectorized SLSQP + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized pairwise check
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    # Only check upper triangle
    iu = np.triu_indices(N, 1)
    if np.any(d2[iu] < rs[iu] ** 2 - tol):
        return False
    return True


def _max_radius_for_circle(cx, cy, r, W, H, i):
    """Max radius for circle i given others fixed."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
    """Check if moving circle i to (new_x, new_y) is valid."""
    if new_x < r[i] or new_x > W - r[i]:
        return False
    if new_y < r[i] or new_y > H - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - cx[j]
        dy = new_y - cy[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-10:
            return False
    return True


def _force_relax(cx, cy, r, W, H, max_iter=2000, time_budget=None):
    """Force-directed relaxation to resolve overlaps and push to boundaries."""
    n = N
    dt = 0.003
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 50
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 50
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 50
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 50

        # Pairwise repulsion
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    overlap = r_sum - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-10:
                        ux = dx / d
                        uy = dy / d
                        f = overlap * 20.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f

        if max_overlap < 1e-9:
            break

        cx = cx + dt * fx
        cy = cy + dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)

    return cx, cy


def _expand_all_radii(cx, cy, r, W, H, time_budget):
    """Iteratively expand each circle's radius to its maximum possible."""
    n = N
    improved = True
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_for_circle(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    return cx, cy, r


def _expansion_polish(cx, cy, r, W, H, time_budget):
    """Iteratively expand radii and shift positions to create space."""
    n = N
    for rnd in range(40):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_for_circle(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            if r_max <= r[i] + 1e-7:
                min_gap = 1e9
                move_dx, move_dy = 0.0, 0.0
                gaps = [
                    (cx[i] - r[i], 1, 0),
                    (W - cx[i] - r[i], -1, 0),
                    (cy[i] - r[i], 0, 1),
                    (H - cy[i] - r[i], 0, -1),
                ]
                for gap, dx_dir, dy_dir in gaps:
                    if gap < min_gap:
                        min_gap = gap
                        move_dx, move_dy = dx_dir, dy_dir
                for j in range(n):
                    if i == j:
                        continue
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        if d > 1e-10:
                            move_dx = dx / d
                            move_dy = dy / d

                if min_gap < 0.03:
                    for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                        new_x = cx[i] + move_dx * step
                        new_y = cy[i] + move_dy * step
                        if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                            cx[i] = new_x
                            cy[i] = new_y
                            improved = True
                            break
                    else:
                        perp_dx, perp_dy = -move_dy, move_dx
                        for step in [0.02, 0.01, 0.005]:
                            for s in [1.0, -1.0]:
                                new_x = cx[i] + s * perp_dx * step
                                new_y = cy[i] + s * perp_dy * step
                                if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                                    cx[i] = new_x
                                    cy[i] = new_y
                                    improved = True
                                    break
                            if improved:
                                break
        if not improved:
            break

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    return cx, cy, r


def _soft_max(x, beta=1000.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if m < -1000:
        return 0.0
    return m + np.log(np.sum(np.exp((x - m) * beta))) / beta


def _vectorized_objective(z, beta=500.0):
    """
    Smooth penalty-based objective for SLSQP using soft-max for min-gap.
    z = [cx, cy, r, W]
    H = 2 - W
    Objective: -sum(r) + penalty for constraint violations
    """
    n = N
    cx = z[0:n]
    cy = z[n:2*n]
    r = z[2*n:3*n]
    W = z[3*n]
    H = 2.0 - W

    # Ensure positive dimensions
    W_safe = max(W, 0.01)
    H_safe = max(H, 0.01)

    penalty = 0.0

    # Wall constraints using soft-max
    wall_viol_x_low = np.maximum(0, r - cx)
    wall_viol_x_high = np.maximum(0, cx + r - W_safe)
    wall_viol_y_low = np.maximum(0, r - cy)
    wall_viol_y_high = np.maximum(0, cy + r - H_safe)

    penalty += 100 * (np.sum(wall_viol_x_low**2) + np.sum(wall_viol_x_high**2) +
                      np.sum(wall_viol_y_low**2) + np.sum(wall_viol_y_high**2))

    # Pairwise distance constraints using soft-max
    # For each pair, violation = max(0, (r_i+r_j)^2 - d^2)
    # We compute all pairwise violations and take soft-max sum
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    viol_matrix = np.maximum(0, rs * rs - d2)
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    viol = viol_matrix[iu]
    penalty += 100 * np.sum(viol ** 2)

    # Ensure radii are positive
    neg_r = np.maximum(0, -r)
    penalty += 1000 * np.sum(neg_r ** 2)

    # Ensure W is in reasonable range
    if W < 0.2 or W > 1.8:
        penalty += 1000 * (W - 1.0) ** 2

    return -np.sum(r) + penalty


def _slsqp_smooth(z0, time_budget, maxiter=500):
    """SLSQP with smooth penalty objective."""
    n = N

    def objective(z):
        return _vectorized_objective(z)

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 2.0), (0.001, 2.0), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})
        z_opt = res.x
        cx_new = z_opt[0:n]
        cy_new = z_opt[n:2*n]
        r_new = z_opt[2*n:3*n]
        W_new = z_opt[-1]
        H_new = 2.0 - W_new
        if _is_valid(cx_new, cy_new, r_new, W_new, H_new):
            return cx_new, cy_new, r_new, W_new, H_new
    except Exception:
        pass
    return None


def _high_precision_polish(cx, cy, r, W, H, time_budget):
    """
    High-precision polish: alternate between position optimization and radius expansion
    with very small steps for fine-tuning.
    """
    n = N
    improved = True
    max_rounds = 200

    while improved and time.time() < time_budget:
        improved = False

        # Try to expand each radius
        for i in range(n):
            r_max = _max_radius_for_circle(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-10:
                lo, hi = r[i], r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    improved = True

        # Try small position adjustments for each circle
        for i in range(n):
            if time.time() > time_budget:
                break
            # Find the most constraining direction
            min_gap = 1e9
            best_dx, best_dy = 0.0, 0.0

            # Check walls
            gaps = [
                (cx[i] - r[i], 1, 0),
                (W - cx[i] - r[i], -1, 0),
                (cy[i] - r[i], 0, 1),
                (H - cy[i] - r[i], 0, -1),
            ]
            for gap, dx_dir, dy_dir in gaps:
                if gap < min_gap:
                    min_gap = gap
                    best_dx, best_dy = dx_dir, dy_dir

            # Check other circles
            for j in range(n):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-10:
                        best_dx = dx / d
                        best_dy = dy / d

            # Try moving in the direction of most constraint (to create space)
            if min_gap < 0.01:
                for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                    new_x = cx[i] + best_dx * step
                    new_y = cy[i] + best_dy * step
                    if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                        cx[i] = new_x
                        cy[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular directions
                    perp_dx, perp_dy = -best_dy, best_dx
                    for step in [0.005, 0.002, 0.001]:
                        for s in [1.0, -1.0]:
                            new_x = cx[i] + s * perp_dx * step
                            new_y = cy[i] + s * perp_dy * step
                            if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                                cx[i] = new_x
                                cy[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

    # Final uniform radius expansion
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    return cx, cy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_cx = best_cy = best_r = None
    best_W = best_H = None

    # Start from 5x5 grid in unit square
    side = 5
    W0, H0 = 1.0, 1.0
    r0 = 0.5 / side * 0.95

    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        cx[i] = (col + 0.5) / side * W0
        cy[i] = (row + 0.5) / side * H0
        r[i] = r0

    cx, cy = _force_relax(cx, cy, r, W0, H0, max_iter=500)
    cx, cy, r = _expand_all_radii(cx, cy, r, W0, H0, deadline - 30)
    val = np.sum(r)
    if _is_valid(cx, cy, r, W0, H0):
        best_value = val
        best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
        best_W, best_H = W0, H0

    # Try multiple aspect ratios
    if time.time() < deadline - 60:
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
        for w0 in aspect_ratios:
            if time.time() > deadline - 50:
                break
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue

            cx_t = np.zeros(N)
            cy_t = np.zeros(N)
            r_t = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx_t[i] = (col + 0.5) / side * w0
                cy_t[i] = (row + 0.5) / side * h0
                r_t[i] = min(w0, h0) / side * 0.9

            cx_t, cy_t = _force_relax(cx_t, cy_t, r_t, w0, h0, max_iter=300)
            cx_t, cy_t, r_t = _expand_all_radii(cx_t, cy_t, r_t, w0, h0, deadline - 40)

            if _is_valid(cx_t, cy_t, r_t, w0, h0):
                val_t = np.sum(r_t)
                if val_t > best_value:
                    best_value = val_t
                    best_cx, best_cy, best_r = cx_t.copy(), cy_t.copy(), r_t.copy()
                    best_W, best_H = w0, h0

                # Try smooth SLSQP on this starting point
                if time.time() < deadline - 30:
                    z0 = np.concatenate([cx_t, cy_t, r_t, [w0]])
                    result = _slsqp_smooth(z0, deadline - 25, maxiter=300)
                    if result is not None:
                        cx_n, cy_n, r_n, W_n, H_n = result
                        val_n = np.sum(r_n)
                        if val_n > best_value:
                            best_value = val_n
                            best_cx, best_cy, best_r = cx_n, cy_n, r_n
                            best_W, best_H = W_n, H_n

    # Final high-precision polish
    if best_cx is not None and time.time() < deadline - 10:
        cx_p, cy_p, r_p = _high_precision_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 5)
        if _is_valid(cx_p, cy_p, r_p, best_W, best_H):
            val_p = np.sum(r_p)
            if val_p > best_value:
                best_value = val_p
                best_cx, best_cy, best_r = cx_p, cy_p, r_p

    if best_cx is None:
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, 5)
            cx[i] = (col + 0.5) / 5
            cy[i] = (row + 0.5) / 5
            r[i] = 0.09
        return np.column_stack((cx, cy, r))

    for _ in range(5):
        r_trial = best_r * (1 - 1e-10)
        if _is_valid(best_cx, best_cy, r_trial, best_W, best_H):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_cx, best_cy, best_r))
