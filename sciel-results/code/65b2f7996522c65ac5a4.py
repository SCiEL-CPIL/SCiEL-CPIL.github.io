import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = np.inf
    # Vectorized approach for speed
    # Get all combinations of 3 indices
    indices = list(combinations(range(N), 3))
    idx = np.array(indices)
    p1 = pts[idx[:, 0]]
    p2 = pts[idx[:, 1]]
    p3 = pts[idx[:, 2]]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Vectorized:
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return np.min(areas)

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Calculate the objective score: min_triangle_area / hull_area."""
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _neg_score(x):
    """Negative score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -np.inf
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Initial candidates
    candidates = []
    # Regular polygon
    candidates.append(_regular_polygon(N, 1.0))
    # Slightly perturbed regular polygons
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
    
    # Global Optimization using Differential Evolution
    # Bounds: points should be within [-2, 2] to allow for scaling
    bounds = [(-2, 2)] * (2 * N)
    
    if time.time() < deadline - 30:
        try:
            # DE is stochastic, set seed for reproducibility in this run
            de_res = differential_evolution(
                _neg_score, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-4, 
                seed=42,
                update_interval=1,
                polish=False  # We will polish manually
            )
            de_pts = de_res.x.reshape(N, 2)
            de_score = _score(de_pts)
            update_best(de_pts, de_score)
        except Exception:
            pass

    # Local Optimization using Nelder-Mead on the best point found
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
            
    # If DE didn't run or failed, try NM on initial best
    if best_points is not None and time.time() < deadline - 5:
         # Try a few more NM runs from different initializations if time permits
         for pts in candidates:
             if time.time() > deadline - 2:
                 break
             try:
                 res = minimize(
                     _neg_score,
                     pts.flatten(),
                     method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
                 )
                 res_pts = res.x.reshape(N, 2)
                 res_score = _score(res_pts)
                 update_best(res_pts, res_score)
             except Exception:
                 pass

    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
