"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Pre-compute all triangle index combinations for vectorized scoring
_tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[_tri_indices[:, 0]]
    pj = p[_tri_indices[:, 1]]
    pk = p[_tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _hull_area(points):
    """Area of convex hull."""
    try:
        h = ConvexHull(points)
        return h.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: min triangle area / hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    return ma / ha


def _objective(flat):
    """Objective to minimize: hull_area / min_triangle_area (inverse of score)."""
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    ma = _min_triangle_area(pts)
    if ma < 1e-12:
        return 1e10
    return ha / ma


def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two concentric rings (7 + 7)
    for r_inner in [0.3, 0.5, 0.7]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        p1 = r_inner * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([np.cos(a2), np.sin(a2)])
        configs.append(np.vstack([p1, p2]))
    
    # 3. Three rings (6 + 4 + 4)
    a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = 0.6 * np.column_stack([np.cos(a2), np.sin(a2)])
    p3 = 0.3 * np.column_stack([np.cos(a3), np.sin(a3)])
    configs.append(np.vstack([p1, p2, p3]))
    
    # 4. Random configurations
    for _ in range(10):
        r = rng.uniform(0, 1, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        configs.append(pts)
        
    # 5. Jittered regular polygon
    base = configs[0]
    for std in [0.05, 0.1, 0.2]:
        jitter = rng.normal(0, std, size=(N, 2))
        configs.append(base + jitter)
        
    return configs


def _refine(pts, time_left):
    """Local refinement using Powell and Nelder-Mead."""
    best_pts = pts.copy()
    best_obj = _objective(best_pts.flatten())
    
    # Powell
    try:
        res = minimize(_objective, best_pts.flatten(), method='Powell',
                       options={'maxiter': int(max(1, time_left * 5)),
                                'ftol': 1e-12, 'xtol': 1e-10})
        if res.fun < best_obj:
            best_obj = res.fun
            best_pts = res.x.reshape(N, 2)
    except Exception:
        pass

    # Nelder-Mead
    try:
        res = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': int(max(1, time_left * 10)),
                                'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_pts = res.x.reshape(N, 2)
    except Exception:
        pass

    return best_pts


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    best_pts = None
    best_score = 0.0
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective(pts.flatten())
    
    # Phase 1: Multi-start local search
    inits = _generate_initial_configs()
    for i, pts0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, pts0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success or res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 2: Differential Evolution (Global Search)
    if time.time() < deadline - 20:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            de_res = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=50, tol=1e-8,
                popsize=10, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        try:
            best_pts = _refine(best_pts, remaining)
            update_best(best_pts)
        except Exception:
            pass
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass
    
    return best_pts
