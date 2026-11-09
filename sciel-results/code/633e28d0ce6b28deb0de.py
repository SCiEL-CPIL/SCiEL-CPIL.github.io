"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_N_TRIS = len(_idx_i)

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _objective(params):
    """Objective function: -min_area + lambda * penalty."""
    points = params.reshape(N, 2)
    pen = _penalty(points)
    area = _min_area(points)
    if pen > 1.0:
        return 1e6 + pen
    return -area + 10.0 * pen

def _coarse_to_fine(points, max_iter=200, time_limit=None, start_time=None):
    """Iteratively optimize the worst triangle's vertices, then try pair moves."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 5.0
    
    pts = points.copy()
    best_area = _min_area(pts)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find the worst triangle
        areas = _all_areas(pts)
        worst_idx = np.argmin(areas)
        worst_area = areas[worst_idx]
        
        if worst_area >= best_area:
            best_area = worst_area
        
        # Get the three points of the worst triangle
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        
        # Try to improve by moving these three points
        # Use a small local search on just these 3 points
        def _sub_objective(sub_params):
            new_pts = pts.copy()
            new_pts[i] = sub_params[0:2]
            new_pts[j] = sub_params[2:4]
            new_pts[k] = sub_params[4:6]
            pen = _penalty(new_pts)
            if pen > 0.1:
                return 1e4 + pen
            return -_min_area(new_pts)
        
        sub_init = np.array([pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1], pts[k, 0], pts[k, 1]])
        
        try:
            res = minimize(
                _sub_objective,
                sub_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < _sub_objective(sub_init):
                new_pts = pts.copy()
                new_pts[i] = res.x[0:2]
                new_pts[j] = res.x[2:4]
                new_pts[k] = res.x[4:6]
                new_area = _min_area(new_pts)
                if new_area > best_area:
                    best_area = new_area
                    pts = new_pts
        except Exception:
            pass
        
        # Try pair-point moves: move pairs of points to increase diversity
        if iteration % 3 == 0:
            # Try moving random pairs
            for _ in range(3):
                a, b = np.random.choice(N, 2, replace=False)
                def _pair_obj(pair_params):
                    new_pts = pts.copy()
                    new_pts[a] = pair_params[0:2]
                    new_pts[b] = pair_params[2:4]
                    pen = _penalty(new_pts)
                    if pen > 0.1:
                        return 1e4 + pen
                    return -_min_area(new_pts)
                
                pair_init = np.array([pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1]])
                try:
                    res_p = minimize(
                        _pair_obj,
                        pair_init,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res_p.fun < _pair_obj(pair_init):
                        new_pts = pts.copy()
                        new_pts[a] = res_p.x[0:2]
                        new_pts[b] = res_p.x[2:4]
                        new_area = _min_area(new_pts)
                        if new_area > best_area:
                            best_area = new_area
                            pts = new_pts
                except Exception:
                    pass
    
    return pts, best_area

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, H)
        
        # Check edge constraints
        if y < 0:
            y = 0
        max_y_1 = np.sqrt(3) * x
        if y > max_y_1:
            y = max_y_1
        max_y_2 = np.sqrt(3) * (1 - x)
        if y > max_y_2:
            y = max_y_2
        
        if x < 1e-9:
            y = 0
        if x > 1 - 1e-9:
            y = 0
            
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    bounds = [(0, 1), (0, H)] * N
    
    best_params = None
    best_score = -np.inf
    
    def _update_best(params):
        nonlocal best_params, best_score
        if params is not None:
            pts = params.reshape(N, 2)
            pen = _penalty(pts)
            if pen < 0.01:
                area = _min_area(pts)
                if area > best_score:
                    best_score = area
                    best_params = params.copy()
    
    # Phase 1: Global search with DE
    try:
        res_de = differential_evolution(
            _objective,
            bounds,
            maxiter=150,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        _update_best(res_de.x)
    except Exception:
        pass
    
    # Phase 2: Global search with dual_annealing
    if time.time() < deadline - 30:
        try:
            res_da = dual_annealing(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                restarts=10,
                polish=False
            )
            _update_best(res_da.x)
        except Exception:
            pass
    
    # Phase 3: Local refinement with coarse-to-fine
    if best_params is not None and time.time() < deadline - 20:
        try:
            refined_pts, refined_area = _coarse_to_fine(
                best_params.reshape(N, 2),
                max_iter=100,
                time_limit=15.0,
                start_time=start_time
            )
            if refined_area > best_score:
                best_score = refined_area
                best_params = refined_pts.flatten()
        except Exception:
            pass
    
    # Phase 4: Additional DE runs with different seeds if time permits
    if time.time() < deadline - 25:
        for seed in [123, 456]:
            if time.time() > deadline - 10:
                break
            try:
                res_de2 = differential_evolution(
                    _objective,
                    bounds,
                    maxiter=80,
                    popsize=25,
                    tol=1e-6,
                    seed=seed,
                    polish=False,
                    init='latinhypercube'
                )
                _update_best(res_de2.x)
                
                # Refine this result
                if time.time() < deadline - 10:
                    refined_pts, refined_area = _coarse_to_fine(
                        res_de2.x.reshape(N, 2),
                        max_iter=50,
                        time_limit=8.0,
                        start_time=start_time
                    )
                    if refined_area > best_score:
                        best_score = refined_area
                        best_params = refined_pts.flatten()
            except Exception:
                pass
    
    # Phase 5: Final polish with Nelder-Mead on the best result
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            _update_best(res_nm.x)
        except Exception:
            pass
    
    # Final fallback
    if best_params is None:
        rng = np.random.default_rng(0)
        pts = np.zeros((N, 2))
        for j in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    pts[j, 0] = v + 0.5 * (1 - u - v)
                    pts[j, 1] = H * (1 - u - v)
                    break
        best_params = pts.flatten()
    
    points = _clamp_to_triangle(best_params.reshape(N, 2))
    return points
