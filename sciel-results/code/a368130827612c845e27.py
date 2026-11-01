"""Circle packing in perimeter-4 rectangle: Multi-phase with contact-graph refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
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


def _force_relax(z, max_iter=500):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

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

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

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

        if max_overlap > 1e-8:
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
        else:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _hexagonal_initial(w, h, n=N):
    """Dense hexagonal grid of circles inside a w x h rectangle."""
    area = w * h
    r_est = np.sqrt(area / (n * 2 * np.sqrt(3)))
    r_est = min(r_est, min(w, h) / 3.0)
    r_est = max(r_est, 0.02)

    x = []
    y = []
    dy = r_est * np.sqrt(3)
    dx = 2 * r_est
    row = 0
    while True:
        y0 = r_est + row * dy
        if y0 + r_est > h:
            break
        offset = (row % 2) * dx / 2
        col = 0
        while True:
            x0 = r_est + offset + col * dx
            if x0 + r_est > w:
                break
            x.append(x0)
            y.append(y0)
            col += 1
        if len(x) >= n + 20:
            break
        row += 1
        if row > 200:
            break

    x = np.array(x)
    y = np.array(y)
    r = np.full_like(x, r_est)

    if len(x) < n:
        side = int(np.ceil(np.sqrt(n)))
        for k in range(n - len(x)):
            row, col = divmod(k, side)
            x = np.append(x, (col + 0.5) / side * w)
            y = np.append(y, (row + 0.5) / side * h)
            r = np.append(r, min(w, h) / side * 0.3)

    return x[:n], y[:n], r[:n]


def _contact_graph_optimize(z, time_budget):
    """
    Identify the contact graph from a feasible packing, then solve a
    system of exact tangency equations to find a more optimal configuration.
    This is the key improvement: we fix which circles touch which, then
    solve the resulting (potentially) convex subproblem.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Identify contact pairs: circles that are very close to touching
    contact_tol = 0.005
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if abs(d - (r[i] + r[j])) < contact_tol:
                contacts.append((i, j))

    # Also identify wall contacts
    wall_contacts = []  # (i, 'left'/'right'/'bottom'/'top')
    for i in range(n):
        if x[i] - r[i] < contact_tol:
            wall_contacts.append((i, 'left'))
        if w - x[i] - r[i] < contact_tol:
            wall_contacts.append((i, 'right'))
        if y[i] - r[i] < contact_tol:
            wall_contacts.append((i, 'bottom'))
        if h - y[i] - r[i] < contact_tol:
            wall_contacts.append((i, 'top'))

    # Build a constraint-based optimization that enforces the contact structure
    # Variables: x_i, y_i, r_i for each circle, plus w
    # We want to maximize sum(r_i) subject to:
    #   - For each contact (i,j): dist(i,j) = r_i + r_j (equality)
    #   - For each wall contact: appropriate equality
    #   - All non-contact pairs: dist(i,j) >= r_i + r_j (inequality)
    #   - All circles inside rectangle

    # Use SLSQP with equality constraints for contacts
    def obj_cg(p):
        return -np.sum(p[2::3][:n])

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    constraints = []

    # Equality constraints for circle-circle contacts
    for (i, j) in contacts:
        constraints.append({
            'type': 'eq',
            'fun': lambda p, i=i, j=j: (
                (p[3 * i] - p[3 * j]) ** 2 +
                (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                (p[3 * i + 2] + p[3 * j + 2]) ** 2
            )
        })

    # Equality constraints for wall contacts
    for (i, side) in wall_contacts:
        if side == 'left':
            constraints.append({
                'type': 'eq',
                'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]
            })
        elif side == 'right':
            constraints.append({
                'type': 'eq',
                'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]
            })
        elif side == 'bottom':
            constraints.append({
                'type': 'eq',
                'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]
            })
        elif side == 'top':
            constraints.append({
                'type': 'eq',
                'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]
            })

    # Inequality constraints for non-contact pairs
    contact_set = set(contacts)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in contact_set:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (
                        (p[3 * i] - p[3 * j]) ** 2 +
                        (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                        (p[3 * i + 2] + p[3 * j + 2]) ** 2
                    )
                })

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(obj_cg, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
        z_opt = res.x
        x_new = z_opt[0::3][:n]
        y_new = z_opt[1::3][:n]
        r_new = z_opt[2::3][:n]
        w_new = z_opt[-1]
        h_new = 2.0 - w_new
        if _is_valid(x_new, y_new, r_new, w_new, h_new):
            return z_opt
    except Exception:
        pass

    return None


def _polish_result(z, time_budget):
    """Polish with smooth gap-aware repulsion + radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(5000, time_budget * 200))
    gap_smooth_len = 0.05

    for it in range(max_iter):
        if time.time() > time_budget:
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

        gap = d_pairs - r_sum_pairs
        overlap_force = np.maximum(-gap, 0.0) * 25.0
        soft_force = 0.8 * np.exp(-gap / gap_smooth_len)
        f_pairs = overlap_force + soft_force

        active = f_pairs > 0.01
        if np.any(active):
            idx_active = np.where(active)[0]
            for idx in idx_active:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                d = d_pairs[idx]
                if d > 1e-6:
                    ux = dx[i_idx, j_idx] / d
                    uy = dy[i_idx, j_idx] / d
                    f = f_pairs[idx]
                    fx[i_idx] += ux * f
                    fy[i_idx] += uy * f
                    fx[j_idx] -= ux * f
                    fy[j_idx] -= uy * f

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0004
        elif max_force > 0.01:
            dt = 0.0008
        else:
            dt = 0.0012

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii uniformly.
    grow_factor = 1.0003
    for _ in range(300):
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


def _expansion_relaxation_polish(z, time_budget):
    """Iteratively expand circles and shift positions to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
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
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True

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

            if min_dist < 0.02:
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

                moved = False
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        moved = True
                        break
                if not moved:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                moved = True
                                break
                        if moved:
                            break

        if not improved:
            break

    grow_factor = 1.001
    for _ in range(100):
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
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
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


def _epigraph_refine(z, time_budget):
    """Smooth log-sum-exp epigraph refinement."""
    z = z.copy().astype(float)
    n = N
    alpha = 40.0

    def soft_violation(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        viol = np.maximum(r_sum[iu] - d_pairs, 0.0)
        v2 = viol * viol
        v2 = np.clip(v2, 1e-12, None)
        m = np.max(v2)
        soft_max = m + np.log(np.sum(np.exp(v2 - m)))
        lv = np.maximum(r - x, 0.0)
        rv = np.maximum(x + r - w, 0.0)
        bv = np.maximum(r - y, 0.0)
        tv = np.maximum(y + r - h, 0.0)
        wall2 = np.concatenate([lv * lv, rv * rv, bv * bv, tv * tv])
        wall2 = np.clip(wall2, 1e-12, None)
        mw = np.max(wall2)
        soft_wall = mw + np.log(np.sum(np.exp(wall2 - mw)))
        return soft_max + soft_wall

    def obj_epi(zv):
        r = zv[2::3][:n]
        return -np.sum(r) + alpha * soft_violation(zv)

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(500, time_budget * 80))
    try:
        res = minimize(obj_epi, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
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


def _local_search_positions(x, y, r, w, h, time_budget):
    """8-direction local search to reposition circles for more space."""
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    improved = True
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
        for i in range(n):
            for angle in angles:
                for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                    new_x = x[i] + step * np.cos(angle)
                    new_y = y[i] + step * np.sin(angle)
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _high_precision_radius_polish(x, y, r, w, h, time_budget):
    """High-precision bisection on each radius to squeeze out maximum gain."""
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    for _ in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
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

            if r_max > r[i] + 1e-12:
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-12:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
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

        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            hx, hy, hr = _hexagonal_initial(w0, h0, N)
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = hx
            z[1::3][:N] = hy
            z[2::3][:N] = hr
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

        z0_relaxed = _force_relax(z0, max_iter=500)

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
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # NEW: Contact graph optimization
        if best_result is not None and time.time() < deadline - 20:
            cg_result = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 10)
            if cg_result is not None:
                valid, val = check_validity(cg_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result

        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

            # Another contact graph pass after expansion
            if best_result is not None and time.time() < deadline - 12:
                cg_result2 = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 5)
                if cg_result2 is not None:
                    valid, val = check_validity(cg_result2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = cg_result2

            if best_result is not None and time.time() < deadline - 10:
                expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
                if expanded2 is not None:
                    valid, val = check_validity(expanded2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded2

        if best_result is not None and time.time() < deadline - 14:
            epi = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 8)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    # --- Phase 3: Sequential placement as alternative approach ---
    if time.time() < deadline - 10:
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            if time.time() > deadline - 5:
                break
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            # Simple grid-based sequential placement
            side = int(np.ceil(np.sqrt(N)))
            x_seq = np.zeros(N)
            y_seq = np.zeros(N)
            r_seq = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x_seq[i] = (col + 0.5) / side * w0
                y_seq[i] = (row + 0.5) / side * h0
                r_seq[i] = min(w0 / side, h0 / side) * 0.4

            z_seq = np.zeros(3 * N + 1)
            z_seq[0::3][:N] = x_seq
            z_seq[1::3][:N] = y_seq
            z_seq[2::3][:N] = r_seq
            z_seq[-1] = w0

            # Polish the sequential result
            seq_polished = _polish_result(z_seq, deadline - (time.time() - start_time) - 3)
            if seq_polished is not None:
                valid, val = check_validity(seq_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = seq_polished

    # --- Phase 4: Radial growth from hexagonal grid ---
    if time.time() < deadline - 8:
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            if time.time() > deadline - 4:
                break
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            hx, hy, hr = _hexagonal_initial(w0, h0, N)
            z_hex = np.zeros(3 * N + 1)
            z_hex[0::3][:N] = hx
            z_hex[1::3][:N] = hy
            z_hex[2::3][:N] = hr
            z_hex[-1] = w0

            # Polish hexagonal grid
            hex_polished = _polish_result(z_hex, deadline - (time.time() - start_time) - 2)
            if hex_polished is not None:
                valid, val = check_validity(hex_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = hex_polished

    # --- Phase 5: Local search + high-precision radius polish ---
    if best_result is not None and time.time() < deadline - 5:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Local search to reposition circles
        ls_result = _local_search_positions(x, y, r, w, h, deadline - (time.time() - start_time) - 3)
        if ls_result is not None:
            lx, ly, lr = ls_result
            z_ls = np.empty(3 * N + 1)
            z_ls[0::3][:N] = lx
            z_ls[1::3][:N] = ly
            z_ls[2::3][:N] = lr
            z_ls[-1] = w
            valid, val = check_validity(z_ls)
            if valid and val > best_value:
                best_value = val
                best_result = z_ls

        # High-precision radius polish
        if best_result is not None:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            hp_result = _high_precision_radius_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            if hp_result is not None:
                hx, hy, hr = hp_result
                z_hp = np.empty(3 * N + 1)
                z_hp[0::3][:N] = hx
                z_hp[1::3][:N] = hy
                z_hp[2::3][:N] = hr
                z_hp[-1] = w
                valid, val = check_validity(z_hp)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_hp

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
