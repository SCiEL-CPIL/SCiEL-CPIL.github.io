"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Centrally symmetric parameterization (exploits known structure for even N).
  2. Support function / radial representation for strictly convex configurations.
  3. Multi-phase optimization: global DE + local NM/Powell/L-BFGS.
  4. Fast vectorized triangle-area computation.
  5. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14
HALF_N = N // 2

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=80.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally symmetric parameterization ----------
# For centrally symmetric configurations, point i and point i+HALF_N are negatives.
# We parameterize by the first HALF_N points (7 points), and the rest are determined.
# Each of the 7 points is (r_i, theta_i) in polar coordinates.
# Total params: 7 * 2 = 14

def _decode_sym(params):
    """Decode 14-dim symmetric params to (N,2) points."""
    r = params[:HALF_N]
    theta = params[HALF_N:]
    
    # Ensure positive radii
    r = np.maximum(r, 0.01)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # First half
    pts_first = np.column_stack([x, y])
    # Second half: negative of first half
    pts_second = -pts_first
    
    return np.vstack([pts_first, pts_second])


def _objective_sym(params):
    pts = _decode_sym(params)
    return _objective_hard(pts.flatten())


def _objective_sym_soft(params, beta=80.0):
    pts = _decode_sym(params)
    return _objective_soft(pts.flatten(), beta)


# ---------- Support function parameterization ----------
# Use sorted angles and radial values for strictly convex configurations.
# This ensures the points are in convex position.

def _support_angles():
    """Generate fixed non-uniform angles clustered near diagonals."""
    # Base uniform angles
    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add some clustering near diagonals (45, 135, 225, 315 degrees)
    # Perturb angles slightly to create non-uniform spacing
    rng = np.random.default_rng(42)
    angles = base + rng.normal(0, 0.05, size=N)
    # Sort to ensure convex position
    angles = np.sort(angles)
    # Normalize to [0, 2*pi)
    angles = angles - angles[0]
    return angles


_SUPPORT_ANGLES = _support_angles()


def _decode_support(params):
    """Decode radial values to (N,2) points using fixed angles."""
    r = np.maximum(params, 0.1)
    x = r * np.cos(_SUPPORT_ANGLES)
    y = r * np.sin(_SUPPORT_ANGLES)
    return np.column_stack([x, y])


def _objective_support(params):
    pts = _decode_support(params)
    return _objective_hard(pts.flatten())


def _objective_support_soft(params, beta=80.0):
    pts = _decode_support(params)
    return _objective_soft(pts.flatten(), beta)


# ---------- Optimizer helpers ----------

def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


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


# ---------- Initial configurations ----------

def _make_sym_inits():
    """Generate diverse initial symmetric parameter vectors."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular symmetric: 7 points on unit circle, evenly spaced
    for r_scale in [0.5, 0.7, 0.85, 1.0, 1.2]:
        r = np.full(HALF_N, r_scale)
        theta = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False)
        params = np.concatenate([r, theta])
        inits.append(params)
    
    # 2. Jittered symmetric
    for js in [0.05, 0.1, 0.15, 0.2]:
        r = np.full(HALF_N, 1.0) + rng.normal(0, js, size=HALF_N)
        r = np.maximum(r, 0.1)
        theta = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False) + rng.normal(0, js, size=HALF_N)
        params = np.concatenate([r, theta])
        inits.append(params)
    
    # 3. Random symmetric
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        r = rs.uniform(0.3, 1.5, size=HALF_N)
        theta = rs.uniform(0, 2 * np.pi, size=HALF_N)
        params = np.concatenate([r, theta])
        inits.append(params)
    
    return inits


def _make_support_inits():
    """Generate diverse initial support parameter vectors."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Uniform radius
    for r_val in [0.5, 0.8, 1.0, 1.2]:
        inits.append(np.full(N, r_val))
    
    # 2. Jittered radius
    for js in [0.05, 0.1, 0.15, 0.2]:
        r = np.full(N, 1.0) + rng.normal(0, js, size=N)
        r = np.maximum(r, 0.1)
        inits.append(r)
    
    # 3. Random radius
    for seed in range(15):
        rs = np.random.default_rng(seed * 17 + 3)
        r = rs.uniform(0.3, 1.5, size=N)
        inits.append(r)
    
    return inits


def _make_cartesian_inits():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Centrally symmetric regular
    for r_scale in [0.7, 0.85, 1.0]:
        angles = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False)
        x = r_scale * np.cos(angles)
        y = r_scale * np.sin(angles)
        pts_first = np.column_stack([x, y])
        pts_second = -pts_first
        inits.append(np.vstack([pts_first, pts_second]))
    
    # 3. Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    
    # 4. Random configurations
    for seed in range(10):
        rs = np.random.default_rng(seed * 17 + 3)
        angles = np.sort(rs.uniform(0, 2 * np.pi, N))
        radii = rs.uniform(0.4, 1.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    # 5. Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    return inits


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

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

    # Phase 1: Optimize in symmetric parameter space (14 dims)
    sym_inits = _make_sym_inits()
    for i, params0 in enumerate(sym_inits):
        if time.time() > deadline - 85:
            break
        res = _nm(params0, deadline, _objective_sym, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _decode_sym(res.x)
            _consider(pts)

    # Phase 2: Optimize in support parameter space (14 dims)
    support_inits = _make_support_inits()
    for i, params0 in enumerate(support_inits):
        if time.time() > deadline - 70:
            break
        res = _nm(params0, deadline, _objective_support, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _decode_support(res.x)
            _consider(pts)

    # Phase 3: Cartesian NM from best and diverse inits
    cart_inits = _make_cartesian_inits()
    if best_pts is not None:
        cart_inits.insert(0, best_pts)

    for i, pts0 in enumerate(cart_inits):
        if time.time() > deadline - 50:
            break
        res = _nm(pts0.flatten(), deadline, _objective_hard, max_iter=1200)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: Powell refinement from best
    if best_pts is not None and time.time() < deadline - 35:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 5: L-BFGS on soft objective
    if best_pts is not None and time.time() < deadline - 25:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=2000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 6: Differential evolution in symmetric space
    if time.time() < deadline - 20:
        try:
            bounds = [(0.1, 2.0)] * HALF_N + [(0, 2 * np.pi)] * HALF_N
            res_de = differential_evolution(
                _objective_sym, bounds,
                seed=42, maxiter=40, tol=1e-10,
                popsize=12, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                pts = _decode_sym(res_de.x)
                _consider(pts)
        except Exception:
            pass

    # Phase 7: Local refinement after DE
    if best_pts is not None and time.time() < deadline - 10:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 8: Powell final polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=600)
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
