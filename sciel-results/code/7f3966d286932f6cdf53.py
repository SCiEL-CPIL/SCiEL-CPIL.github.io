"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Use a log-barrier constrained optimization where the 28 Cartesian coordinates
     are penalized for violating strict convexity (negative cross products of
     consecutive hull edges) and collinearity.
  2. This allows the optimizer to explore non-radial, asymmetric convex polygons
     that radial parameterizations cannot represent.
  3. Multi-phase: global (DE) + local (NM, Powell, L-BFGS) refinement.
  4. Time-guarded to finish within 120s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


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


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _convexity_violation(points):
    """
    Compute convexity violation for a set of points.
    Returns 0 if points are in strictly convex position (sorted CCW),
    otherwise returns a positive measure of violation.
    """
    n = points.shape[0]
    # Sort points by angle around centroid
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    order = np.argsort(angles)
    pts_sorted = points[order]
    
    # Compute cross products of consecutive edges
    # Edge i goes from pts_sorted[i] to pts_sorted[(i+1)%n]
    # Cross product should be positive for CCW convex polygon
    total_violation = 0.0
    for i in range(n):
        p0 = pts_sorted[i]
        p1 = pts_sorted[(i + 1) % n]
        p2 = pts_sorted[(i + 2) % n]
        # Vector p0->p1 and p1->p2
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            total_violation += -cross
    
    return total_violation


def _objective_hard(flat, mu=100.0):
    """
    Objective: hull_area / min_triangle_area + mu * convexity_violation
    Minimize this.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Check convexity
    conv_viol = _convexity_violation(pts)
    if conv_viol > 1e-6:
        return 1e10 + mu * conv_viol
    
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    return hull_area / min_area


def _objective_soft(flat, beta=50.0, mu=100.0):
    """
    Soft objective using log-sum-exp for min area.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Check convexity
    conv_viol = _convexity_violation(pts)
    if conv_viol > 1e-4:
        return 1e10 + mu * conv_viol
    
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

def _make_initial_points():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)

    # 3. Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)

    # 4. Random configurations
    for seed in range(15):
        rs = np.random.default_rng(seed * 17 + 3)
        angles = np.sort(rs.uniform(0, 2 * np.pi, N))
        radii = rs.uniform(0.4, 1.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # 5. Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))

    # 6. Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))

    # 7. Asymmetric configurations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 7)
        # Random perturbation of regular polygon with varying magnitude
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angle_jitter = rs.normal(0, 0.15, size=N)
        radius_jitter = rs.normal(1.0, 0.15, size=N)
        thetas = base_angles + angle_jitter
        radii = np.abs(radius_jitter) + 0.2
        pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
        inits.append(pts)

    return inits


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        # Verify convexity
        if _convexity_violation(pts) > 1e-6:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Local optimization from diverse initial points
    cart_inits = _make_initial_points()
    for i, pts0 in enumerate(cart_inits):
        if time.time() > deadline - 80:
            break
        res = _nm(pts0.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 2: Powell refinement from best
    if best_pts is not None and time.time() < deadline - 50:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=600)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: L-BFGS on soft objective
    if best_pts is not None and time.time() < deadline - 40:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: Differential evolution (global search) if time permits
    if time.time() < deadline - 30:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            res_de = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=30, tol=1e-10,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                _consider(res_de.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 5: Local refinement after DE
    if best_pts is not None and time.time() < deadline - 20:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 6: Powell final polish
    if best_pts is not None and time.time() < deadline - 12:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 7: Final L-BFGS polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1000)
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
