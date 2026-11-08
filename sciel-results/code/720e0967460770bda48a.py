"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples for vectorized area computation
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles using vectorized ops."""
    pi = points[_ci]
    pj = points[_cj]
    pk = points[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle (vectorized)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum(y, 0.0)
    upper = np.minimum(H * x, H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area(flat_pts):
    """Objective for minimization: negative of minimum triangle area."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _max_min_distance_init(rng, n_iter=2000):
    """Initialize points by maximizing minimum pairwise distance (greedy + random restarts)."""
    best_pts = None
    best_min_dist = 0.0
    
    for _ in range(n_iter):
        # Start with a random point
        pts = [rng.uniform(0, 1, 2)]
        pts = [_project_to_triangle(np.array(pts)).flatten()]
        
        for i in range(1, N):
            # Greedy: place next point to maximize min distance to existing points
            best_candidate = None
            best_dist = 0.0
            
            for _ in range(500):  # Try 500 random candidates
                candidate = rng.uniform(0, 1, 2)
                candidate = _project_to_triangle(np.array([candidate])).flatten()
                
                # Compute min distance to existing points
                dists = np.linalg.norm(np.array(pts) - candidate, axis=1)
                min_dist = np.min(dists)
                
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_candidate = candidate
            
            pts.append(best_candidate)
        
        pts = np.array(pts)
        
        # Compute min pairwise distance
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_pts = pts.copy()
    
    return best_pts


def _local_optimize_nm(pts, max_time=60.0):
    """Run local optimization using Nelder-Mead starting from pts."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _neg_min_area, flat_pts, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Get a good initialization via max-min distance
    t0 = time.time()
    init_pts = _max_min_distance_init(rng, n_iter=500)
    init_score = _min_triangle_area(init_pts)
    
    best_points = init_pts.copy()
    best_score = init_score
    
    # Phase 2: Nelder-Mead from the max-min init
    if time.time() < deadline - 10:
        opt_pts, opt_score = _local_optimize_nm(best_points, max_time=40.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 3: Try a few random restarts with Nelder-Mead
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation of best
        perturbed = best_points + rng.normal(0, 0.05, size=(N, 2))
        perturbed = _project_to_triangle(perturbed)
        
        opt_pts, opt_score = _local_optimize_nm(perturbed, max_time=30.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 4: Final SLSQP polish
    if time.time() < deadline - 10:
        flat_pts = best_points.flatten()
        try:
            result = minimize(
                _neg_min_area, flat_pts, method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_points
