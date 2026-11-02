"""Circle packing in perimeter-4 rectangle: SLSQP + Voronoi expansion + smooth refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21


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


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _max_radius_simple(x, y, r, w, h, i):
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


def _max_radius_voronoi(x, y, r, w, h, i):
    pts = list(zip(x, y))
    boundary_pts = [
        (0, 0), (w, 0), (0, h), (w, h),
        (0, h / 2), (w, h / 2), (w / 2, 0), (w / 2, h),
        (w * 0.25, 0), (w * 0.75, 0), (w * 0.25, h), (w * 0.75, h),
        (0, h * 0.25), (0, h * 0.75), (w, h * 0.25), (w, h * 0.75),
    ]
    all_pts = np.array(pts + boundary_pts)

    try:
        vor = Voronoi(all_pts)
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            return _max_radius_simple(x, y, r, w, h, i)

        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return _max_radius_simple(x, y, r, w, h, i)

        min_dist = float('inf')
        for k in range(len(vertices)):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % len(vertices)]
            dist = _point_to_segment_dist(x[i], y[i], v1[0], v1[1], v2[0], v2[1])
            if dist < min_dist:
                min_dist = dist

        r_voronoi = min_dist
    except Exception:
        return _max_radius_simple(x, y, r, w, h, i)

    r_max = r_voronoi
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

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
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


def _smooth_objective(z, penalty_weight=500.0, soft_beta=50.0):
    """Smooth surrogate using soft-min for distance constraints."""
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
    # Soft overlap: use log-sum-exp style smooth max
    violation = r_sum_pairs - d_pairs
    # Smooth max via log-sum-exp
    max_v = np.max(violation)
    if max_v > -1.0 / soft_beta:
        smooth_max = max_v + np.log(np.mean(np.exp(soft_beta * (violation - max_v)))) / soft_beta
        obj += penalty_weight * smooth_max
    else:
        obj += penalty_weight * np.sum(np.maximum(violation, 0.0) ** 2)

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

    return obj


def _force_relax(z, max_iter=800):
    """Fully vectorized force-directed relaxation."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    iu = np.triu_indices(n, k=1)

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

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)
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
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _voronoi_expansion_polish(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
            else:
                min_gap = float('inf')
                blocker_idx = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        blocker_idx = j

                wall_gaps = [x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]]
                min_wall_gap = min(wall_gaps)

                if min_gap < 0.015 or min_wall_gap < 0.015:
                    directions = []
                    if blocker_idx >= 0:
                        dx = x[i] - x[blocker_idx]
                        dy = y[i] - y[blocker_idx]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-8:
                            directions.append((dx / d, dy / d))

                    if wall_gaps[0] < 0.015:
                        directions.append((1.0, 0.0))
                    if wall_gaps[1] < 0.015:
                        directions.append((-1.0, 0.0))
                    if wall_gaps[2] < 0.015:
                        directions.append((0.0, 1.0))
                    if wall_gaps[3] < 0.015:
                        directions.append((0.0, -1.0))

                    for dx_dir, dy_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        norm = np.sqrt(2.0)
                        directions.append((dx_dir / norm, dy_dir / norm))

                    for dx_dir, dy_dir in directions:
                        for step in [0.03, 0.015, 0.008, 0.004]:
                            new_x = x[i] + dx_dir * step
                            new_y = y[i] + dy_dir * step
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
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00002
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _vectorized_polish(z, time_budget):
    """Fully vectorized force-directed polish with adaptive step."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    dt = 0.001
    max_iter = int(min(5000, (time_budget - time.time()) * 200))

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

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-12:
            f_pairs = overlap * 25.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.0003
        elif max_force > 0.1:
            dt = 0.0008
        else:
            dt = 0.0015

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        if max_overlap < 1e-10 and max_force < 0.005:
            break

    # Uniform radius growth
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00005
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00001
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _smooth_refine(z, time_budget):
    """SLSQP with smooth log-sum-exp objective to escape local minima."""
    z = z.copy().astype(float)
    n = N

    def obj_smooth(zv):
        return _smooth_objective(zv, penalty_weight=300.0, soft_beta=40.0)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    maxiter = int(min(500, (time_budget - time.time()) * 100))

    try:
        res = minimize(obj_smooth, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h, tol=1e-9):
            return z_opt
        # Try slight shrink
        r_shrunk = r * 0.999
        if _is_valid(x, y, r_shrunk, w, h, tol=1e-9):
            z_opt[2::3][:n] = r_shrunk
            return z_opt
    except Exception:
        pass
    return None


def _constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high precision."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        return -np.sum(zv[2::3][:n])

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        bi = 3 * i
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi: p[bi] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi: p[-1] - p[bi] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi: p[bi + 1] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi: (2.0 - p[-1]) - p[bi + 1] - p[bi + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = 3 * i, 3 * j
            constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi, bj=bj:
                                (p[bi] - p[bj]) ** 2 + (p[bi + 1] - p[bj + 1]) ** 2 -
                                (p[bi + 2] + p[bj + 2]) ** 2})

    maxiter = int(min(400, (time_budget - time.time()) * 80))
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

        if _is_valid(x, y, r, w, h, tol=1e-9):
            return z_opt
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

        # Grid-based starts with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 1.22, 0.78]
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

        # Hexagonal packing
        for w0 in [0.9, 1.0, 1.1, 1.2]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 5
            cols = 5
            spacing_x = w0 / (cols + 0.5)
            spacing_y = h0 / (rows + 0.5)
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
                cy = (row + 0.5) * spacing_y
                r0 = min(spacing_x, spacing_y) * 0.48
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for seed in range(15):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
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

        z0_relaxed = _force_relax(z0, max_iter=600)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(150, remaining * 40)))
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

    # --- Phase 2: Polish pipeline ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Vectorized force-directed polish
        polished = _vectorized_polish(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Voronoi-based expansion
        if best_result is not None and time.time() < deadline - 20:
            expanded = _voronoi_expansion_polish(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Smooth log-sum-exp SLSQP refinement (escape local minima)
        if best_result is not None and time.time() < deadline - 15:
            refined = _smooth_refine(best_result, deadline - (time.time() - start_time) - 8)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Constraint-based SLSQP
        if best_result is not None and time.time() < deadline - 10:
            refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined2 is not None:
                valid, val = check_validity(refined2)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined2

        # 5. Final vectorized polish
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _vectorized_polish(best_result, deadline - (time.time() - start_time))
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

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
