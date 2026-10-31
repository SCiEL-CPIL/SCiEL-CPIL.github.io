"""Circle packing in perimeter-4 rectangle: PBD-inspired iterative projection + SLSQP + polish."""
import numpy as np
import time
from scipy.optimize import minimize, fsolve

N = 21


def _soft_max(v, beta=200.0):
    m = np.maximum(v, 0.0)
    return m + np.logaddexp(0.0, beta * (v - m)) / beta


def _vectorized_objective(z, penalty_weight=1000.0):
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
    overlap = _soft_max(r_sum_pairs - d_pairs, beta=200.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = _soft_max(r - x, beta=200.0)
    right_viol = _soft_max(x + r - w, beta=200.0)
    bottom_viol = _soft_max(r - y, beta=200.0)
    top_viol = _soft_max(y + r - h, beta=200.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) + np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) + np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    return np.all(d_sq[iu] >= r_sum[iu] ** 2 - tol)


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
            return False
    return True


def _pbd_project(x, y, r, w, h, alpha=1.0):
    """
    Position-Based Dynamics projection: iteratively project circle centers
    to satisfy non-overlap and wall constraints. Each circle has target radius r_i.
    Returns (x_new, y_new) with all constraints satisfied (within tolerance).
    """
    n = N
    x = x.copy()
    y = y.copy()
    # Clamp to walls
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    max_iter = 200
    for it in range(max_iter):
        # Wall projection
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Circle-circle projection
        converged = True
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        # Find violations
        need_fix = r_sum_pairs > np.sqrt(np.maximum(d_sq_pairs, 1e-14)) + 1e-12
        if not np.any(need_fix):
            break

        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            ddx = x[i_idx] - x[j_idx]
            ddy = y[i_idx] - y[j_idx]
            d = np.sqrt(ddx * ddx + ddy * ddy)
            r_s = r[i_idx] + r[j_idx]

            if d < r_s - 1e-12:
                if d < 1e-12:
                    # Circles at same center; push apart arbitrarily
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = ddx / d, ddy / d
                # Move each circle half the overlap (weighted by inverse radius for stability)
                overlap = (r_s - d) * alpha * 0.5
                # Weighted by radius ratio (heavier circles move less)
                wi = r[j_idx] / (r[i_idx] + r[j_idx])
                wj = r[i_idx] / (r[i_idx] + r[j_idx])
                x[i_idx] += ux * overlap * 2.0 * wi
                y[i_idx] += uy * overlap * 2.0 * wi
                x[j_idx] -= ux * overlap * 2.0 * wj
                y[j_idx] -= uy * overlap * 2.0 * wj
                converged = False

        # Re-clamp to walls after circle projections
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        if converged:
            break

    return x, y


def _pbd_expand(x, y, r, w, h, max_rounds=80):
    """
    PBD-style expansion: try to grow each circle's radius, then project
    positions to maintain non-overlap. Iterate until no further growth possible.
    Includes local repositioning to find better expansion opportunities.
    """
    n = N
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    small_steps = [0.02, 0.01, 0.005, 0.002]

    for rnd in range(max_rounds):
        improved = False
        for i in range(n):
            # Compute max possible radius for circle i
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
            r_max = max(0.01, r_max)

            # Try local repositioning to find better r_max
            best_rmax = r_max
            best_pos = None
            for dx_d, dy_d in directions:
                for s in small_steps:
                    nx = x[i] + dx_d * s
                    ny = y[i] + dy_d * s
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    rw = min(nx, w - nx, ny, h - ny)
                    rm = rw
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        ra = d - r[j]
                        if ra < rm:
                            rm = ra
                    rm = max(0.01, rm)
                    if rm > best_rmax:
                        best_rmax = rm
                        best_pos = (nx, ny)

            if best_pos is not None and best_rmax > r_max + 1e-8:
                x[i], y[i] = best_pos
                r_max = best_rmax

            if r_max > r[i] + 1e-10:
                # Try to expand
                new_r = min(r_max, r[i] * 1.005 + 0.001)
                r[i] = new_r
                # Project to fix overlaps
                x, y = _pbd_project(x, y, r, w, h)
                if _is_valid(x, y, r, w, h):
                    improved = True
                else:
                    r[i] /= 1.005
                    r[i] -= 0.001
                    x, y = _pbd_project(x, y, r, w, h)

        if not improved:
            break

    return x, y, r


def _pbd_position_search(x, y, r, w, h, time_budget, max_trials=300):
    """
    PBD-inspired position search: try random perturbations and project,
    looking for configurations that allow larger total radii.
    """
    n = N
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = float(np.sum(r))
    rng = np.random.default_rng(2024)

    for trial in range(max_trials):
        if time.time() > time_budget:
            break

        x_p = best_x.copy()
        y_p = best_y.copy()
        r_p = best_r.copy()

        # Perturb random subset of circles
        n_perturb = rng.integers(2, 9)
        indices = rng.choice(n, size=n_perturb, replace=False)
        for idx in indices:
            angle = rng.uniform(0, 2 * np.pi)
            step = rng.uniform(0.005, 0.04)
            new_x = x_p[idx] + step * np.cos(angle)
            new_y = y_p[idx] + step * np.sin(angle)
            if _is_valid_move(x_p, y_p, r_p, w, h, idx, new_x, new_y):
                x_p[idx] = new_x
                y_p[idx] = new_y

        # Project
        x_p, y_p = _pbd_project(x_p, y_p, r_p, w, h)

        # Try to expand
        x_p, y_p, r_p = _pbd_expand(x_p, y_p, r_p, w, h, max_rounds=40)

        if _is_valid(x_p, y_p, r_p, w, h):
            val = float(np.sum(r_p))
            if val > best_val + 1e-10:
                best_val = val
                best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()

    return best_x, best_y, best_r


def _contact_graph_solve(z, time_budget):
    """Solve exact tangency equations for identified contact structure."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    contact_tol = 0.008
    cc_contacts = []
    cw_contacts = []

    for i in range(n):
        if x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 0))
        if w - x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 1))
        if y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 2))
        if h - y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 3))
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d - (r[i] + r[j]) < contact_tol:
                cc_contacts.append((i, j))

    n_eq = len(cc_contacts) + len(cw_contacts)
    if n_eq < 3 * n - 2:
        contact_tol = 0.03
        cc_contacts = []
        cw_contacts = []
        for i in range(n):
            if x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 0))
            if w - x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 1))
            if y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 2))
            if h - y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 3))
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d - (r[i] + r[j]) < contact_tol:
                    cc_contacts.append((i, j))
        n_eq = len(cc_contacts) + len(cw_contacts)

    if n_eq < 3 * n - 2:
        return None

    def equations(vars):
        vx = vars[:n]
        vy = vars[n:2*n]
        vr = vars[2*n:3*n]
        eqs = []
        for i, j in cc_contacts:
            dx = vx[i] - vx[j]
            dy = vy[i] - vy[j]
            eqs.append(dx * dx + dy * dy - (vr[i] + vr[j]) ** 2)
        for i, wall in cw_contacts:
            if wall == 0:
                eqs.append(vx[i] - vr[i])
            elif wall == 1:
                eqs.append(vx[i] + vr[i] - w)
            elif wall == 2:
                eqs.append(vy[i] - vr[i])
            elif wall == 3:
                eqs.append(vy[i] + vr[i] - h)
        return eqs

    vars0 = np.concatenate([x, y, r])

    try:
        sol, info, ier, msg = fsolve(equations, vars0, full_output=True, xtol=1e-13, maxfev=8000)
        if ier != 1:
            rng = np.random.default_rng(7)
            for attempt in range(3):
                if time.time() > time_budget:
                    break
                perturbed = vars0 + rng.normal(0, 0.005, size=vars0.shape)
                for i in range(n):
                    perturbed[i] = np.clip(perturbed[i], 0.01, w - 0.01)
                    perturbed[n + i] = np.clip(perturbed[n + i], 0.01, h - 0.01)
                    perturbed[2*n + i] = np.clip(perturbed[2*n + i], 0.01, min(w, h) * 0.4)
                sol, info, ier, msg = fsolve(equations, perturbed, full_output=True, xtol=1e-13, maxfev=5000)
                if ier == 1:
                    break

        if ier == 1:
            sol_x, sol_y, sol_r = sol[:n], sol[n:2*n], sol[2*n:3*n]
            if _is_valid(sol_x, sol_y, sol_r, w, h, tol=1e-7):
                val = float(np.sum(sol_r))
                if val > float(np.sum(r)) + 1e-8:
                    z_new = np.empty(3 * N + 1)
                    z_new[0::3][:n] = sol_x
                    z_new[1::3][:n] = sol_y
                    z_new[2::3][:n] = sol_r
                    z_new[-1] = w
                    return z_new
            else:
                for factor in [0.999, 0.995, 0.99]:
                    sol_r_s = sol_r * factor
                    if _is_valid(sol_x, sol_y, sol_r_s, w, h, tol=1e-7):
                        if float(np.sum(sol_r_s)) > float(np.sum(r)) + 1e-8:
                            z_new = np.empty(3 * N + 1)
                            z_new[0::3][:n] = sol_x
                            z_new[1::3][:n] = sol_y
                            z_new[2::3][:n] = sol_r_s
                            z_new[-1] = w
                            return z_new
                        break
    except Exception:
        pass

    return None


def _final_polish(x, y, r, w, h, time_budget):
    """Final high-precision polish: per-circle binary search expansion + local moves with gap-aware repulsion."""
    n = N
    # More directions including finer angles
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    steps = [0.04, 0.025, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

    for outer in range(100):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Max radius at current position
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

            # Try small moves for better r_max - multi-scale search
            best_rmax = r_max
            best_pos = None
            for dx_d, dy_d in directions:
                for s in steps:
                    nx = x[i] + dx_d * s
                    ny = y[i] + dy_d * s
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    rw = min(nx, w - nx, ny, h - ny)
                    rm = rw
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        ra = d - r[j]
                        if ra < rm:
                            rm = ra
                    rm = max(0.0, rm)
                    if rm > best_rmax:
                        best_rmax = rm
                        best_pos = (nx, ny)

            if best_pos is not None and best_rmax > r_max + 1e-9:
                x[i], y[i] = best_pos
                r_max = best_rmax

            # Binary search expand
            if r_max > r[i] + 1e-10:
                lo, hi = r[i], r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_t = r.copy()
                    r_t[i] = mid
                    if _is_valid(x, y, r_t, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Gap-aware repulsion polish: push non-contact circles apart to create space
    for _ in range(200):
        if time.time() > time_budget:
            break
        moved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Compute clearance to each neighbor
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                # If gap is small but positive, try to increase it
                if 0 < gap < 0.02:
                    # Try moving i away from j slightly
                    if d > 1e-12:
                        ux, uy = dx / d, dy / d
                    else:
                        continue
                    for step in [0.003, 0.0015, 0.0008]:
                        nx = x[i] + ux * step
                        ny = y[i] + uy * step
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        valid = True
                        for k in range(n):
                            if k == i:
                                continue
                            ddx = nx - x[k]
                            ddy = ny - y[k]
                            if ddx * ddx + ddy * ddy < (r[i] + r[k]) ** 2 - 1e-9:
                                valid = False
                                break
                        if valid:
                            x[i], y[i] = nx, ny
                            moved = True
                            break

        if not moved:
            break

    # Uniform expansion
    for _ in range(800):
        if time.time() > time_budget:
            break
        r_t = r * 1.0001
        if _is_valid(x, y, r_t, w, h):
            r = r_t
        else:
            break

    return x, y, r


def _slsqp_with_constraints(z, time_budget):
    """SLSQP with explicit inequality constraints (boundary + pairwise)."""
    n = N

    def obj(zv):
        return -np.sum(zv[2::3][:n])

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def make_oc(idx1, idx2):
                return lambda p: (p[3*idx1] - p[3*idx2])**2 + (p[3*idx1+1] - p[3*idx2+1])**2 - (p[3*idx1+2] + p[3*idx2+2])**2
            constraints.append({'type': 'ineq', 'fun': make_oc(i, j)})

    maxiter = int(min(500, max(100, time_budget * 80)))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 80})
        x = res.x[0::3][:n]
        y = res.x[1::3][:n]
        r = res.x[2::3][:n]
        w = res.x[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return res.x
    except Exception:
        pass
    return None


def _force_relax(z, max_iter=300):
    """Force-directed relaxation with smooth gap-aware repulsion to create space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.004

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

        # Also apply gentle repulsion for near-contact pairs to create space
        gap = d_pairs - r_sum_pairs
        near_contact = (gap > 0) & (gap < 0.03)
        repulsion = np.where(near_contact, (0.03 - gap) * 2.0, 0.0)

        if np.max(overlap) < 1e-8 and np.max(repulsion) < 1e-6:
            break

        f_pairs = overlap * 15.0 + repulsion
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        for idx in range(len(iu[0])):
            ii = iu[0][idx]
            jj = iu[1][idx]
            fx[ii] += ux[idx] * f_pairs[idx]
            fy[ii] += uy[idx] * f_pairs[idx]
            fx[jj] -= ux[idx] * f_pairs[idx]
            fy[jj] -= uy[idx] * f_pairs[idx]

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3*i] = np.clip(z[3*i], r[i], w - r[i])
            z[3*i+1] = np.clip(z[3*i+1], r[i], h - r[i])

    return z


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if _is_valid(x, y, r, w, h, tol):
            return True, float(np.sum(r))
        return False, 0.0

    # Generate initial guesses
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
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        # PBD projection to resolve overlaps
        x = z0[0::3][:N].copy()
        y = z0[1::3][:N].copy()
        r = z0[2::3][:N].copy()
        w = z0[-1]
        h = 2.0 - w
        x, y = _pbd_project(x, y, r, w, h)
        z0_pbd = np.empty(3*N+1)
        z0_pbd[0::3][:N] = x
        z0_pbd[1::3][:N] = y
        z0_pbd[2::3][:N] = r
        z0_pbd[-1] = w

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(lambda z: _vectorized_objective(z), z0_pbd, method='SLSQP',
                           bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            pass

    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 30:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # 2a: PBD expansion
        if time.time() < deadline - 20:
            x, y, r = _pbd_expand(x, y, r, w, h, max_rounds=60)

        # 2b: PBD position search
        if time.time() < deadline - 15:
            x, y, r = _pbd_position_search(x, y, r, w, h, deadline - (time.time() - start_time) - 12)

        # 2c: Contact graph
        if time.time() < deadline - 10:
            z_cur = np.empty(3*N+1)
            z_cur[0::3][:N] = x
            z_cur[1::3][:N] = y
            z_cur[2::3][:N] = r
            z_cur[-1] = w
            cg = _contact_graph_solve(z_cur, deadline - (time.time() - start_time) - 8)
            if cg is not None:
                valid, val = check_validity(cg)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg
                    x = cg[0::3][:N].copy()
                    y = cg[1::3][:N].copy()
                    r = cg[2::3][:N].copy()

        # 2d: SLSQP with constraints
        if time.time() < deadline - 6:
            z_cur = np.empty(3*N+1)
            z_cur[0::3][:N] = x
            z_cur[1::3][:N] = y
            z_cur[2::3][:N] = r
            z_cur[-1] = w
            slsqp_res = _slsqp_with_constraints(z_cur, deadline - (time.time() - start_time) - 4)
            if slsqp_res is not None:
                valid, val = check_validity(slsqp_res)
                if valid and val > best_value:
                    best_value = val
                    best_result = slsqp_res
                    x = slsqp_res[0::3][:N].copy()
                    y = slsqp_res[1::3][:N].copy()
                    r = slsqp_res[2::3][:N].copy()

        # 2e: Final polish
        if time.time() < deadline - 2:
            x, y, r = _final_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            z_final = np.empty(3*N+1)
            z_final[0::3][:N] = x
            z_final[1::3][:N] = y
            z_final[2::3][:N] = r
            z_final[-1] = w
            valid, val = check_validity(z_final)
            if valid and val > best_value:
                best_value = val
                best_result = z_final

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    for _ in range(5):
        r_t = r * (1 - 1e-10)
        if _is_valid(x, y, r_t, w, h):
            r = r_t
        else:
            break

    return np.column_stack((x, y, r))
