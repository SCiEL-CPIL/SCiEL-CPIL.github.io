"""Circle packing in perimeter-4 rectangle: SLSQP + Enhanced Polish."""
import numpy as np
import time
from scipy.optimize import minimize

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


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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
    if new_x < r[i] - 1e-9 or new_x > w - r[i] + 1e-9:
        return False
    if new_y < r[i] - 1e-9 or new_y > h - r[i] + 1e-9:
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


def _force_relax(z, max_iter=500):
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


def _polish_positions(x, y, r, w, h, time_budget, dt=0.001, wall_k=35, pair_k=18, max_iter=3000):
    n = N
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_k
        fx += right_viol * wall_k
        fy -= bottom_viol * wall_k
        fy += top_viol * wall_k

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * pair_k
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
            dt_use = 0.0005
        else:
            dt_use = dt

        x += dt_use * fx
        y += dt_use * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _expand_radii(x, y, r, w, h, time_budget):
    n = N
    # Uniform expansion
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Per-circle expansion with position optimization
    for _ in range(80):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            # Try to find better position for circle i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_now = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_now:
                    r_max_now = r_allowed
            r_max_now = max(0.0, r_max_now)

            if r_max_now > r[i] + 1e-6:
                # Try to move circle i to a position that allows larger radius
                best_pos = None
                best_r = r_max_now
                # Sample positions on a grid around current position
                for dx_m in np.linspace(-0.05, 0.05, 11):
                    for dy_m in np.linspace(-0.05, 0.05, 11):
                        nx = x[i] + dx_m
                        ny = y[i] + dy_m
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        if not _is_valid_move(x, y, r, w, h, i, nx, ny):
                            continue
                        rw = min(nx, w - nx, ny, h - ny)
                        rm = rw
                        for j in range(n):
                            if i == j:
                                continue
                            dx2 = nx - x[j]
                            dy2 = ny - y[j]
                            d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
                            ra = d2 - r[j]
                            if ra < rm:
                                rm = ra
                        rm = max(0.0, rm)
                        if rm > best_r + 1e-6:
                            best_r = rm
                            best_pos = (nx, ny)

                if best_pos is not None:
                    x[i], y[i] = best_pos
                    r_max = _max_radius_for_circle(x, y, r, w, h, i)
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
                        if lo > r[i] + 1e-8:
                            r[i] = lo
                            improved = True

        if not improved:
            break

    return x, y, r


def _pairwise_optimize(x, y, r, w, h, time_budget):
    """Optimize positions of pairs of touching circles to create more space."""
    n = N
    for _ in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > time_budget:
                    break
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < 0.03:  # Only optimize close pairs
                    # Try to move both circles slightly to increase gap
                    for di in np.linspace(-0.01, 0.01, 5):
                        for dj in np.linspace(-0.01, 0.01, 5):
                            # Move i in direction away from j
                            if d > 1e-6:
                                ux = dx / d
                                uy = dy / d
                            else:
                                ux, uy = 1.0, 0.0
                            ni_x = x[i] + ux * di
                            ni_y = y[i] + uy * di
                            nj_x = x[j] - ux * dj
                            nj_y = y[j] - uy * dj

                            if ni_x < r[i] or ni_x > w - r[i] or ni_y < r[i] or ni_y > h - r[i]:
                                continue
                            if nj_x < r[j] or nj_x > w - r[j] or nj_y < r[j] or nj_y > h - r[j]:
                                continue

                            # Check validity
                            valid = True
                            for k in range(n):
                                if k == i or k == j:
                                    continue
                                dx_ki = ni_x - x[k]
                                dy_ki = ni_y - y[k]
                                if dx_ki * dx_ki + dy_ki * dy_ki < (r[i] + r[k]) ** 2 - 1e-9:
                                    valid = False
                                    break
                                dx_kj = nj_x - x[k]
                                dy_kj = nj_y - y[k]
                                if dx_kj * dx_kj + dy_kj * dy_kj < (r[j] + r[k]) ** 2 - 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                continue

                            new_d = np.sqrt((ni_x - nj_x) ** 2 + (ni_y - nj_y) ** 2)
                            new_gap = new_d - (r[i] + r[j])
                            if new_gap > gap + 1e-6:
                                x[i], y[i] = ni_x, ni_y
                                x[j], y[j] = nj_x, nj_y
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        if not improved:
            break
    return x, y


def _constraint_refine(z, time_budget):
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
        h = z_opt[-1] * 0 + 2.0 - z_opt[-1]

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _multi_start_local_search(z_best, time_budget):
    n = N
    best_val = 0.0
    best_z = None

    x0 = z_best[0::3][:n].copy()
    y0 = z_best[1::3][:n].copy()
    r0 = z_best[2::3][:n].copy()
    w0 = z_best[-1]
    h0 = 2.0 - w0

    rng = np.random.default_rng(123)
    perturbations = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]

    for pert in perturbations:
        if time.time() > time_budget:
            break

        x = x0 + rng.normal(0, pert, size=n)
        y = y0 + rng.normal(0, pert, size=n)
        r = r0.copy()

        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)

        # Quick force-directed relaxation
        x, y = _polish_positions(x, y, r, w0, h0, time_budget, dt=0.002, wall_k=30, pair_k=15, max_iter=200)

        # Expand radii
        x, y, r = _expand_radii(x, y, r, w0, h0, time_budget)

        if _is_valid(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:n] = x
                z_new[1::3][:n] = y
                z_new[2::3][:n] = r
                z_new[-1] = w0
                best_z = z_new.copy()

    if best_z is None and _is_valid(x0, y0, r0, w0, h0):
        best_val = float(np.sum(r0))
        best_z = z_best.copy()

    return best_z


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

    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 25:
        # 1. Position polish
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        x, y = _polish_positions(x, y, r, w, h, deadline - (time.time() - start_time) - 15,
                                  dt=0.001, wall_k=35, pair_k=18, max_iter=3000)

        # 2. Pairwise optimization
        x, y = _pairwise_optimize(x, y, r, w, h, deadline - (time.time() - start_time) - 10)

        # 3. Expand radii
        x, y, r = _expand_radii(x, y, r, w, h, deadline - (time.time() - start_time) - 8)

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:N] = x
            z_new[1::3][:N] = y
            z_new[2::3][:N] = r
            z_new[-1] = w
            valid, val = check_validity(z_new)
            if valid and val > best_value:
                best_value = val
                best_result = z_new

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final tight polish
        if best_result is not None and time.time() < deadline - 4:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y = _polish_positions(x, y, r, w, h, deadline - (time.time() - start_time) - 2,
                                      dt=0.0003, wall_k=50, pair_k=25, max_iter=8000)
            x, y, r = _expand_radii(x, y, r, w, h, deadline - (time.time() - start_time))

            if _is_valid(x, y, r, w, h):
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r
                z_new[-1] = w
                valid, val = check_validity(z_new)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_new

        # 6. Multi-start local search
        if best_result is not None and time.time() < deadline - 2:
            best_local = _multi_start_local_search(best_result, deadline - (time.time() - start_time) - 0.5)
            if best_local is not None:
                valid, val = check_validity(best_local)
                if valid and val > best_value:
                    best_value = val
                    best_result = best_local

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
