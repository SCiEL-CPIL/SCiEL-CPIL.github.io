import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _radial_to_cartesian(r, theta):
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _cartesian_to_radial(points):
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    return r, theta


def _soft_min(values, temp):
    if temp <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temp)))
    return min_val - temp * log_sum


def _penalty_obj_radial(x, mu, temp):
    """
    x = [t, r1..rN, theta1..thetaN]
    Minimize: -t/hull_area + mu * sum(max(0, t - area_i)^2)
    Uses soft-min for smoothness when temp > 0.
    """
    t = x[0]
    r = np.clip(x[1:1 + N], 1e-6, 2.0)
    theta = x[1 + N:1 + 2 * N]
    pts = _radial_to_cartesian(r, theta)
    
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(pts)
    
    # Penalty for violations
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # Use soft-min for smoother gradient
    if temp > 1e-12:
        sm = _soft_min(areas, temp)
    else:
        sm = float(np.min(areas))
    
    return -t / hull_a + mu * penalty


def _optimize_radial_penalty(points, deadline, mu_steps=12, max_iter=100):
    """
    Penalty-based optimization in radial coordinates with L-BFGS-B.
    """
    best = points.copy()
    best_s = _score(best)
    
    r0, theta0 = _cartesian_to_radial(best)
    r0 = np.clip(r0, 1e-6, 2.0)
    
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.8
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:1 + N] = r0
    x0[1 + N:] = theta0
    
    mus = np.geomspace(5.0, 1e6, mu_steps)
    temps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.0]
    
    for i, mu in enumerate(mus):
        if time.time() > deadline - 6:
            break
        temp = temps[min(i, len(temps) - 1)]
        try:
            res = minimize(
                lambda x: _penalty_obj_radial(x, mu, temp),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            x0 = res.x
            r = np.clip(x0[1:1 + N], 1e-6, 2.0)
            theta = x0[1 + N:]
            cand = _radial_to_cartesian(r, theta)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Final polish with very high mu
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _penalty_obj_radial(x, 1e7, 0.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1 + N], 1e-6, 2.0)
            theta = x0[1 + N:]
            cand = _radial_to_cartesian(r, theta)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _optimize_cartesian_powell(points, deadline):
    """Polish in Cartesian with Powell."""
    best = points.copy()
    best_s = _score(best)
    
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass
    
    return best, best_s


def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_centered():
    """Center + inner ring (6) + outer ring (6)."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
    for i in range(6):
        a = 2 * np.pi * i / 6 + np.pi / 6
        pts[7 + i] = [1.0 * np.cos(a), 1.0 * np.sin(a)]
    return pts


def _seed_golden_angle():
    """Golden angle spiral-like distribution on unit disk."""
    pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = 2 * np.pi * i * (1 + np.sqrt(5)) / 2
        pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_seed_regular())
    seeds.append(_seed_centered())
    seeds.append(_seed_golden_angle())
    
    # Two-ring seeds
    for n_out in [5, 6, 7, 8]:
        for r_in in [0.4, 0.5, 0.6, 0.7]:
            for a_off in [0.0, np.pi / n_out]:
                seeds.append(_seed_two_ring(n_out, r_in, a_off))
    
    # Three-ring seeds
    for (n1, n2, n3, r2, r3, o2, o3) in [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.5, 0.3, np.pi / 6, 0.0),
    ]:
        seeds.append(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    # Score all seeds
    for seed in seeds:
        _update(seed)
    
    # Primary optimization: radial penalty from best seed
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _optimize_radial_penalty(best_points, deadline - 15, mu_steps=10, max_iter=80)
            _update(pts)
        except Exception:
            pass
    
    # Secondary: optimize from other promising seeds
    for seed in seeds[:8]:
        if time.time() > deadline - 20:
            break
        s_seed = _score(seed)
        if s_seed >= best_score * 0.95:
            try:
                pts, s = _optimize_radial_penalty(seed, deadline - 10, mu_steps=8, max_iter=60)
                _update(pts)
            except Exception:
                pass
    
    # Tertiary: Cartesian Powell polish from best
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _optimize_cartesian_powell(best_points, deadline - 4)
            _update(pts)
        except Exception:
            pass
    
    # Final: another radial penalty pass from current best
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _optimize_radial_penalty(best_points, deadline - 2, mu_steps=6, max_iter=150)
            _update(pts)
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
