"""Circle packing in perimeter-4 rectangle: multi-start SLSQP + Voronoi polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid (vectorized)."""
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
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-14:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _max_radius_simple(x, y, r, w, h, i):
    """Compute max radius for circle i by distance to walls and other circles."""
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
    """Compute maximum radius for circle i using Voronoi cell inradius."""
    r_simple = _max_radius_simple(x, y, r, w, h, i)

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
            return r_simple

        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return r_simple

        min_dist = float('inf')
        for k in range(len(vertices)):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % len(vertices)]
            dist = _point_to_segment_dist(x[i], y[i], v1[0], v1[1], v2[0], v2[1])
            if dist < min_dist:
                min_dist = dist

        r_voronoi = min_dist
    except Exception:
        return r_simple

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


def _objective(z, penalty_weight=2000.0):
    """Objective with quadratic penalties for overlaps and boundary violations."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
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

    # Aspect ratio soft constraint
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _smooth_objective(z, penalty_weight=1500.0, beta=30.0):
    """Smooth objective using log-sum-exp for differentiable penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    violations = r_sum[iu] - d_pairs

    # Smooth penalty via log-sum-exp
    v_max = np.max(violations)
    if v_max > 0:
        v_shifted = violations - v_max
        smooth_max = v_max + (1.0 / beta) * np.log(np.sum(np.exp(v_shifted)))
        obj += penalty_weight * smooth_max * v_max
    else:
        obj += penalty_weight * 0.5 * np.sum(violations * violations)

    # Boundary penalties (smooth)
    for v in [r - x, x + r - w, r - y, y + r - h]:
        v_max = np.max(v)
        if v_max > 0:
            v_shifted = v - v_max
            sm = v_max + (1.0 / beta) * np.log(np.sum(np.exp(v_shifted)))
            obj += penalty_weight * sm * v_max
        else:
            obj += penalty_weight * 0.5 * np.sum(v * v)

    # Aspect ratio soft constraint
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(z, max_iter=400):
    """Force-directed relaxation to resolve initial overlaps."""
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

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

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

        if max_overlap > 1e-8:
            f_pairs = overlap * 12.0
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

        if max_overlap < 1e-8:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(0.0005, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.008, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _voronoi_expansion_polish(z, time_budget):
    """
    Polish using Voronoi-based radius computation.
    Iteratively expands each circle to its Voronoi-limited maximum.
    """
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
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(20):
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

        if not improved:
            break

    # Final uniform expansion
    for _ in range(200):
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


def _polish_result(z, time_budget):
    """Polish with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0008
    max_iter = int(min(2000, time_budget * 120))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 30
        fx += right_viol * 30
        fy -= bottom_viol * 30
        fy += top_viol * 30

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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0004
        else:
            dt = 0.0008

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0002
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


def _high_precision_polish(z, time_budget):
    """
    High-precision polish: iterative radius growth with position adjustment.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Force-directed position refinement
    dt = 0.0004
    max_iter = int(min(4000, time_budget * 200))

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
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-12:
            f_pairs = overlap * 25.0
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
        if max_force > 0.5:
            dt = 0.0002
        elif max_force > 0.1:
            dt = 0.0004
        else:
            dt = 0.0008

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        if max_overlap < 1e-10 and max_force < 0.01:
            break

    # Phase 2: Voronoi-guided individual radius expansion
    for _ in range(40):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-9:
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
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Phase 3: Final simultaneous squeeze
    for _ in range(300):
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


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using smooth log-sum-exp objective."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        violations = r_sum[iu] - d_pairs

        v_max = np.max(violations)
        if v_max > 0:
            v_shifted = violations - v_max
            smooth_max = v_max + (1.0 / 35.0) * np.log(np.sum(np.exp(v_shifted)))
            obj += 500.0 * smooth_max * v_max
        else:
            obj += 500.0 * np.sum(np.maximum(violations, 0.0) ** 2)

        for v in [r - x, x + r - w, r - y, y + r - h]:
            v_max = np.max(v)
            if v_max > 0:
                v_shifted = v - v_max
                sm = v_max + (1.0 / 35.0) * np.log(np.sum(np.exp(v_shifted)))
                obj += 500.0 * sm * v_max
            else:
                obj += 500.0 * np.sum(np.maximum(v, 0.0) ** 2)

        return obj

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    maxiter = int(min(600, time_budget * 120))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - z_opt[-1]

        if not _is_valid(x, y, r, w, h):
            for _ in range(20):
                r = r * 0.999
                if _is_valid(x, y, r, w, h):
                    z_opt[2::3][:n] = r
                    break
            else:
                return None

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _epigraph_radius_growth(z, time_budget):
    """
    Epigraph-based radius growth: solve for max t such that all constraints hold
    with radii scaled by t.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    scale = 0.995
    r_base = r * scale

    def epigraph_obj(v):
        xv = v[0:n]
        yv = v[n:2*n]
        t = v[2*n]
        if t < 0.1 or t > 2.0:
            return 1e6
        rv = r_base * t

        obj = -t * np.sum(r_base)

        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        violations = r_sum[iu] - d_pairs
        penalty = np.sum(np.maximum(violations, 0.0) ** 2) * 600.0
        obj += penalty

        left_v = np.maximum(rv - xv, 0.0)
        right_v = np.maximum(xv + rv - w, 0.0)
        bottom_v = np.maximum(rv - yv, 0.0)
        top_v = np.maximum(yv + rv - h, 0.0)
        obj += 600.0 * (np.sum(left_v**2) + np.sum(right_v**2) +
                        np.sum(bottom_v**2) + np.sum(top_v**2))

        return obj

    bounds = [(0, 2)] * (2 * n) + [(0.5, 1.5)]
    v0 = np.concatenate([x, y, [1.0]])

    maxiter = int(min(400, time_budget * 80))
    try:
        res = minimize(epigraph_obj, v0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 80})

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2*n]
        t_opt = v_opt[2*n]
        r_opt = r_base * t_opt

        if _is_valid(x_opt, y_opt, r_opt, w, h, tol=1e-9):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x_opt
            z_new[1::3][:n] = y_opt
            z_new[2::3][:n] = r_opt
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

        # Grid-based with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 0.9, 1.1, 0.95, 1.05]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random with Poisson-disk-like placement
        rng = np.random.default_rng(42)
        for seed in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            placed = []
            for i in range(N):
                r0 = rng.uniform(0.05, 0.11)
                attempts = 0
                while attempts < 60:
                    cx = rng.uniform(r0, w0 - r0)
                    cy = rng.uniform(r0, h0 - r0)
                    ok = True
                    for (px, py, pr) in placed:
                        if (cx - px)**2 + (cy - py)**2 < (r0 + pr + 0.008)**2:
                            ok = False
                            break
                    if ok:
                        placed.append((cx, cy, r0))
                        z[3 * i] = cx
                        z[3 * i + 1] = cy
                        z[3 * i + 2] = r0
                        break
                    attempts += 1
                else:
                    row, col = divmod(i, side)
                    cx = (col + 0.5) / side * w0
                    cy = (row + 0.5) / side * h0
                    r0 = min(w0 / side, h0 / side) * 0.38
                    z[3 * i] = cx
                    z[3 * i + 1] = cy
                    z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal packing
        for w0 in [0.9, 1.0, 1.1, 1.05, 0.95]:
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
                r0 = min(spacing_x, spacing_y) * 0.46
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Multi-start SLSQP optimization
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        z0_relaxed = _force_relax(z0, max_iter=250)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(350, max(80, remaining * 35)))
        try:
            res = minimize(_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})

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
        # 1. Force-directed polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Voronoi-based expansion
        if best_result is not None and time.time() < deadline - 15:
            expanded = _voronoi_expansion_polish(best_result, deadline - (time.time() - start_time) - 8)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Epigraph-based radius growth
        if best_result is not None and time.time() < deadline - 10:
            epi = _epigraph_radius_growth(best_result, deadline - (time.time() - start_time) - 5)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 6:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 3)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final high-precision polish
        if best_result is not None and time.time() < deadline - 3:
            final_polished = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if final_polished is not None:
                valid, val = check_validity(final_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_polished

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

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
