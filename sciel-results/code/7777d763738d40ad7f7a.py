"""Circle packing in perimeter-4 rectangle: SLSQP + LSE + Constraint Polish + Adaptive Local Search + Greedy Squeeze."""
import numpy as np
import time
from scipy.optimize import minimize

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

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


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


def _force_relax_unified(z, max_iter=400):
    """Fast force-directed relaxation to resolve initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.008
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


def _adaptive_local_search(x, y, r, w, h, time_budget):
    """Adaptive 16-direction local search with geometric step decay."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N

    # 16 directions
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    step = 0.02
    min_step = 1e-6
    decay = 0.7

    for rnd in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            best_pos = None
            best_gain = 0.0

            # Current radius
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

            current_r = min(r[i], r_max)

            # Try moving in each direction
            for d in dirs:
                nx = x[i] + step * d[0]
                ny = y[i] + step * d[1]

                # Check bounds
                if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                    continue

                # Compute new max radius
                r_wall_new = min(nx, w - nx, ny, h - ny)
                r_max_new = r_wall_new
                for j in range(n):
                    if i == j:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    d_dist = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d_dist - r[j]
                    if r_allowed < r_max_new:
                        r_max_new = r_allowed
                r_max_new = max(0.0, r_max_new)

                new_r = min(r[i], r_max_new)

                if new_r > current_r + 1e-9:
                    if best_pos is None or new_r > best_gain:
                        best_gain = new_r
                        best_pos = (nx, ny)

            if best_pos is not None:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                # Recompute radius
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
                r[i] = min(r[i], max(0.0, r_max))
                improved = True

        if not improved:
            step *= decay
            if step < min_step:
                break

    return x, y, r


def _greedy_expand_circles(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N

    for rnd in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
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

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    return x, y, r


def _per_circle_tight_expand(x, y, r, w, h, time_budget):
    """Final per-circle expansion with tight tolerance for squeezing out remaining gains."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N

    for rnd in range(10):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
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

            if r_max > r[i] + 1e-10:
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    return x, y, r


def _polish_result(z, time_budget):
    """Polish with force-directed adjustment + adaptive local search + greedy radius expansion."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Adaptive local search
    x, y, r = _adaptive_local_search(x, y, r, w, h, time_budget)

    # Greedy radius expansion
    x, y, r = _greedy_expand_circles(x, y, r, w, h, time_budget)

    # Second force-directed pass after expansion
    dt2 = 0.001
    max_iter2 = int(min(1000, time_budget * 50))
    for it in range(max_iter2):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt2 = 0.0005
        elif max_force > 0.01:
            dt2 = 0.001
        else:
            dt2 = 0.002

        x_new = x + dt2 * fx
        y_new = y + dt2 * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Final greedy expansion
    x, y, r = _greedy_expand_circles(x, y, r, w, h, time_budget)

    # Uniform expansion
    for _ in range(50):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
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
    """SLSQP with explicit inequality constraints and tighter tolerance."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            result = dist - (r[i] + r)
            result[i] = 1.0
            return result
        return constraint

    def make_wall_constraint(axis, side):
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w
            if axis == 'x':
                pos = x
            else:
                pos = y
            if side == 'left' or side == 'bottom':
                return pos - r
            else:
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 300})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.9999, 0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            # Post-SLSQP per-circle expansion
            x, y, r = _per_circle_tight_expand(x, y, r, w, h, time_budget)
            z_opt[0::3][:n] = x
            z_opt[1::3][:n] = y
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _lse_refine(z, time_budget):
    """Smooth LSE surrogate refinement with tighter tolerance."""
    z = z.copy().astype(float)
    n = N

    def lse_obj(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        viol = r_sum_pairs - d_pairs

        wall_viol = np.concatenate([
            np.maximum(r - x, 0.0),
            np.maximum(x + r - w, 0.0),
            np.maximum(r - y, 0.0),
            np.maximum(y + r - h, 0.0),
        ])

        beta = 300.0
        all_viol = np.concatenate([viol, wall_viol])
        m = np.max(all_viol)
        lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta

        return -np.sum(r) + 50.0 * lse

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(lse_obj, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.9999, 0.999, 0.998, 0.997, 0.995, 0.99, 0.98]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            # Post-LSE per-circle expansion
            x, y, r = _per_circle_tight_expand(x, y, r, w, h, time_budget)
            z_opt[0::3][:n] = x
            z_opt[1::3][:n] = y
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _squeeze_polish(z, time_budget):
    """Iterative squeeze: adaptive local search + greedy expand + short SLSQP reposition, repeated."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    for rnd in range(10):
        if time.time() > time_budget:
            break

        # Adaptive local search
        x, y, r = _adaptive_local_search(x, y, r, w, h, time_budget)

        # Greedy expand each circle
        x, y, r = _greedy_expand_circles(x, y, r, w, h, time_budget)

        # Short SLSQP to reposition
        z_curr = np.empty(3 * N + 1)
        z_curr[0::3][:n] = x
        z_curr[1::3][:n] = y
        z_curr[2::3][:n] = r
        z_curr[-1] = w

        def obj_sq(zv):
            rv = zv[2::3][:n]
            return -np.sum(rv)

        def make_c(i):
            def c(zv):
                xv = zv[0::3][:n]
                yv = zv[1::3][:n]
                rv = zv[2::3][:n]
                dx = xv[i] - xv
                dy = yv[i] - yv
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                res = dist - (rv[i] + rv)
                res[i] = 1.0
                return res
            return c

        def make_wall(ax, sd):
            def c(zv):
                xv = zv[0::3][:n]
                yv = zv[1::3][:n]
                rv = zv[2::3][:n]
                wv = zv[-1]
                hv = 2.0 - wv
                pos = xv if ax == 'x' else yv
                if sd in ('left', 'bottom'):
                    return pos - rv
                else:
                    return (wv if ax == 'x' else hv) - pos - rv
            return c

        cons = [{'type': 'ineq', 'fun': make_c(i)} for i in range(n)]
        cons.append({'type': 'ineq', 'fun': make_wall('x', 'left')})
        cons.append({'type': 'ineq', 'fun': make_wall('x', 'right')})
        cons.append({'type': 'ineq', 'fun': make_wall('y', 'bottom')})
        cons.append({'type': 'ineq', 'fun': make_wall('y', 'top')})

        bnds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

        try:
            res = minimize(obj_sq, z_curr, method='SLSQP',
                           bounds=bnds, constraints=cons,
                           options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 150})
            z_new = res.x
            x_new = z_new[0::3][:n]
            y_new = z_new[1::3][:n]
            r_new = z_new[2::3][:n]
            w_new = z_new[-1]
            h_new = 2.0 - w_new

            if _is_valid(x_new, y_new, r_new, w_new, h_new):
                x, y, r, w, h = x_new, y_new, r_new, w_new, h_new
            else:
                break
        except Exception:
            break

    # Final tight expansion
    x, y, r = _per_circle_tight_expand(x, y, r, w, h, time_budget)

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

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

    # --- Phase 2: Multi-pass refinement ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Force-directed polish + adaptive local search + greedy expansion
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 25)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. LSE smooth refine with tighter tolerance
        if best_result is not None and time.time() < deadline - 25:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 15)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        # 3. Constraint-based SLSQP refine with tighter tolerance
        if best_result is not None and time.time() < deadline - 18:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 10)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Greedy squeeze (adaptive local search + expand + reposition iteratively)
        if best_result is not None and time.time() < deadline - 12:
            squeezed = _squeeze_polish(best_result, deadline - (time.time() - start_time) - 6)
            if squeezed is not None:
                valid, val = check_validity(squeezed)
                if valid and val > best_value:
                    best_value = val
                    best_result = squeezed

        # 5. Second LSE pass (exploit gains from squeeze)
        if best_result is not None and time.time() < deadline - 8:
            lse2 = _lse_refine(best_result, deadline - (time.time() - start_time) - 4)
            if lse2 is not None:
                valid, val = check_validity(lse2)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse2

        # 6. Final constraint refine
        if best_result is not None and time.time() < deadline - 5:
            refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined2 is not None:
                valid, val = check_validity(refined2)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined2

        # 7. Final polish + greedy expansion + tight per-circle expansion
        if best_result is not None and time.time() < deadline - 2:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

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

    # Final safety: ensure strict validity with tight per-circle expansion
    x, y, r = _per_circle_tight_expand(x, y, r, w, h, time.time() + 1.0)

    for _ in range(30):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
