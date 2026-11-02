"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Use a sorted-angle (convex position) parameterization to enforce convexity.
  2. Use a radial parameterization r_i(θ_i) for the boundary.
  3. Multi-phase optimization: global (DE) + local (NM, Powell) refinement.
  4. Time-guarded to finish within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(__import__('itertools').combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    n = points.shape[0]
    p = points
    # For each triangle (i,j,k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Vectorize over all triangles
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
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


def _objective_soft(flat, beta=50.0):
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


# ---------- Convex position parameterization ----------
# Points are parameterized by sorted angles θ_0 < θ_1 < ... < θ_{N-1} on [0, 2π)
# and radii r_i > 0. This guarantees convex position.

def _decode_convex_params(params):
    """
    params: [θ_0, ..., θ_{N-1}, r_0, ..., r_{N-1}]
    θ_i are in [0, 2π), sorted. r_i > 0.
    Returns (N, 2) array.
    """
    thetas = params[:N]
    radii = params[N:]
    # Sort thetas and corresponding radii
    order = np.argsort(thetas)
    thetas_sorted = thetas[order]
    radii_sorted = radii[order]
    x = radii_sorted * np.cos(thetas_sorted)
    y = radii_sorted * np.sin(thetas_sorted)
    return np.column_stack([x, y])


def _encode_convex_params(points):
    """Encode points into convex position parameters [θ_0,...,θ_{N-1}, r_0,...,r_{N-1}]."""
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    thetas = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    radii = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    order = np.argsort(thetas)
    return np.concatenate([thetas[order], radii[order]])


def _objective_convex(params):
    """Objective for convex position parameters."""
    pts = _decode_convex_params(params)
    return _objective_hard(pts.flatten())


def _objective_convex_soft(params, beta=50.0):
    """Soft objective for convex position parameters."""
    pts = _decode_convex_params(params)
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

def _make_initial_convex_params():
    """Generate diverse initial convex position parameter configurations."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    inits.append(np.concatenate([angles, radii]))

    # 2. Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        thetas = np.sort(np.concatenate([a1, a2]))
        radii = np.concatenate([np.full(7, r1), np.ones(7)])
        # Re-sort radii to match thetas
        order = np.argsort(np.concatenate([a1, a2]))
        radii_sorted = np.concatenate([np.full(7, r1), np.ones(7)])[order]
        inits.append(np.concatenate([thetas, radii_sorted]))

    # 3. Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        all_thetas = np.concatenate([a1, a2, a3])
        all_radii = np.concatenate([np.full(5, r1), np.full(4, r2), np.ones(5)])
        order = np.argsort(all_thetas)
        inits.append(np.concatenate([all_thetas[order], all_radii[order]]))

    # 4. Random convex configurations
    for seed in range(20):
        rs = np.random.default_rng(seed * 17 + 3)
        thetas = np.sort(rs.uniform(0, 2 * np.pi, N))
        radii = rs.uniform(0.4, 1.2, N)
        inits.append(np.concatenate([thetas, radii]))

    # 5. Jittered regular polygon
    base_thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    for js in [0.05, 0.1, 0.2]:
        thetas = np.sort(base_thetas + rng.normal(0, js, size=N))
        radii = 1.0 + rng.normal(0, js * 0.5, size=N)
        radii = np.abs(radii) + 0.1
        inits.append(np.concatenate([thetas, radii]))

    # 6. Elliptical configurations (approximated in polar)
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = rx * np.cos(a)
        y = ry * np.sin(a)
        thetas = np.arctan2(y, x)
        radii = np.sqrt(x**2 + y**2)
        order = np.argsort(thetas)
        inits.append(np.concatenate([thetas[order], radii[order]]))

    return inits


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

    # Phase 1: Optimize in convex position parameter space
    convex_inits = _make_initial_convex_params()
    for i, params0 in enumerate(convex_inits):
        if time.time() > deadline - 80:
            break
        res = _nm(params0, deadline, _objective_convex, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _decode_convex_params(res.x)
            _consider(pts)

    # Phase 2: Optimize in Cartesian space from best
    cart_inits = _make_initial_points()
    if best_pts is not None:
        cart_inits.insert(0, best_pts)

    for i, pts0 in enumerate(cart_inits):
        if time.time() > deadline - 50:
            break
        res = _nm(pts0.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: Powell refinement from best
    if best_pts is not None and time.time() < deadline - 40:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=600)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: L-BFGS on soft objective
    if best_pts is not None and time.time() < deadline - 30:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 5: Differential evolution (global search) if time permits
    if time.time() < deadline - 25:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            res_de = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=25, tol=1e-10,
                popsize=12, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                _consider(res_de.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 6: Local refinement after DE
    if best_pts is not None and time.time() < deadline - 15:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 7: Powell final polish
    if best_pts is not None and time.time() < deadline - 8:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 8: Final L-BFGS polish
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
