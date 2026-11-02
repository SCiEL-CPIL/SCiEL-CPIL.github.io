"""Heilbronn convex regions n=14 — combined structural pipeline + refinement machinery."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points (returns min area, not doubled)."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft(points_flat, beta=150.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def _hull_pull(points, alpha=0.5):
    """Expand convex hull outward (structural move to enlarge hull area, improving ratio)."""
    try:
        hull = ConvexHull(points)
        hull_idx = set(hull.vertices.tolist())
    except Exception:
        return points.copy()
    centroid = points.mean(axis=0)
    new_pts = points.copy()
    for idx in hull_idx:
        v = points[idx] - centroid
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            new_pts[idx] = points[idx] + alpha * v
    return new_pts

def _best_of_two(a, b):
    sa = _score(a)
    sb = _score(b)
    return (a, sa) if sa >= sb else (b, sb)

def _nm(x0, deadline, max_iter=2500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
        return res
    except Exception:
        return None

def _coordinate_descent(pts, deadline, max_rounds=60):
    """Cyclic coordinate descent on the hard objective (cheap, different local path)."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 6:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 6:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(28):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-9:
                        break
        if not improved:
            break
    return pts

# ---------- Initializations ----------

def _make_inits():
    inits = []
    rng = np.random.default_rng(2024)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg.flatten())

    # Rotated regular polygons
    for rot in [0.1, 0.4, 0.9, 1.3, 2.0, 3.5]:
        a = angles + rot
        inits.append(np.column_stack([np.cos(a), np.sin(a)]).flatten())

    # Two concentric rings (7 inner, 7 outer) — key structural family
    for r_inner in [0.3, 0.45, 0.6, 0.75, 0.85]:
        a_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
        pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
        inits.append(np.vstack([pts_in, pts_out]).flatten())

    # 3 rings (5, 5, 4)
    for r1, r2 in [(0.3, 0.65), (0.4, 0.8), (0.5, 0.9)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        inits.append(np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ]).flatten())

    # Interleaved circles (two radii, alternating assignment)
    for r1, r2 in [(0.55, 1.0), (0.7, 1.0), (0.85, 1.0)]:
        rad = np.array([r1, r2] * 7)
        inits.append(np.column_stack([rad * np.cos(angles), rad * np.sin(angles)]).flatten())

    # Arc grids (points spread over a band of angles with varying radius)
    for seed in range(4):
        rs = np.random.default_rng(seed * 11 + 5)
        ang = np.sort(rs.uniform(0, 2 * np.pi, N))
        rad = rs.uniform(0.6, 1.15, N)
        inits.append(np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]).flatten())

    # Jittered regular polygon
    for js in [0.05, 0.1, 0.18, 0.3, 0.45]:
        inits.append((reg + rng.normal(0, js, size=(N, 2))).flatten())

    # Random on circle with varying radius
    for seed in range(8):
        rs = np.random.default_rng(seed * 7 + 3)
        ang = np.sort(rs.uniform(0, 2 * np.pi, N))
        rad = rs.uniform(0.5, 1.3, N)
        inits.append(np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]).flatten())

    # Random in unit square
    for seed in range(6):
        rs = np.random.default_rng(seed * 13 + 7)
        inits.append(rs.uniform(-1, 1, size=(N, 2)).flatten())

    return inits

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under the 150s wall-clock kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    inits = _make_inits()

    # Phase 1: Nelder-Mead on hard objective from diverse starts
    for x0 in inits:
        if time.time() > deadline - 30:
            break
        res = _nm(x0, deadline, max_iter=1800, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 2: soft-min NM from best (smoother landscape)
    if best_pts is not None and time.time() < deadline - 35:
        res = _nm(best_pts.flatten(), deadline, max_iter=2500, obj=_objective_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: hull-pull structural move + refine
    if best_pts is not None and time.time() < deadline - 40:
        for alpha in [0.1, 0.2, 0.35, 0.5, 0.7]:
            if time.time() > deadline - 25:
                break
            pulled = _hull_pull(best_pts, alpha=alpha)
            _consider(pulled)
            res = _nm(pulled.flatten(), deadline, max_iter=1200, obj=_objective_hard)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider(res.x.reshape(N, 2))

    # Phase 4: coordinate descent (different local refinement path)
    if best_pts is not None and time.time() < deadline - 30:
        cd = _coordinate_descent(best_pts, deadline, max_rounds=60)
        _consider(cd)

    # Phase 5: Powell polish on the best
    if best_pts is not None and time.time() < deadline - 25:
        res = _powell(best_pts.flatten(), deadline, max_iter=2000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 6: L-BFGS on soft objective (gradient-based, finer convergence on smooth landscape)
    if best_pts is not None and time.time() < deadline - 20:
        res = _lbfgs(best_pts.flatten(), deadline, max_iter=5000, obj=_objective_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 7: coordinate descent again after L-BFGS (fill spare time)
    if best_pts is not None and time.time() < deadline - 15:
        cd = _coordinate_descent(best_pts, deadline, max_rounds=40)
        _consider(cd)

    # Phase 8: final NM + soft polish with tighter tolerances
    if best_pts is not None and time.time() < deadline - 15:
        res = _nm(best_pts.flatten(), deadline, max_iter=3000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))
        if time.time() < deadline - 10:
            res = _nm(best_pts.flatten(), deadline, max_iter=2500, obj=_objective_soft)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider(res.x.reshape(N, 2))

    # Phase 9: final Powell with very tight tolerance
    if best_pts is not None and time.time() < deadline - 8:
        res = _powell(best_pts.flatten(), deadline, max_iter=3000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
