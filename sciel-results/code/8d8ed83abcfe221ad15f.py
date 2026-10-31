import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _compute_triangle_areas(pts):
    """Compute all triangle areas."""
    n = len(pts)
    idx = np.array(list(combinations(range(n), 3)))
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _soft_min_area(areas, alpha=20.0):
    """Soft minimum of areas."""
    # Avoid overflow
    min_a = np.min(areas)
    # If all areas are 0, return 0
    if min_a < 1e-12:
        return 0.0
    # Shift to prevent overflow
    shifted = areas - min_a
    # sum exp(-alpha * shifted)
    # Since shifted >= 0, exp(-alpha * shifted) <= 1
    sum_exp = np.sum(np.exp(-alpha * shifted))
    # soft_min = min_a - (1/alpha) * log(sum_exp)
    return min_a - (1.0 / alpha) * np.log(sum_exp)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_hard(pts):
    """Hard score for final evaluation."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score_soft(params, alpha=20.0):
    """Negative soft score for optimization."""
    pts = params.reshape(N, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0 # Or a large penalty? 0 is bad for minimization (we want small negative). 
        # Actually, if hull area is 0, score is 0, neg score is 0. 
        # We want to minimize neg score. If score is 0, neg score is 0.
        # But valid configurations have score > 0.
        # So 0 is "worse" than -0.1.
        # So returning 0 is a valid penalty for degenerate hulls.
        
    areas = _compute_triangle_areas(pts)
    soft_min = _soft_min_area(areas, alpha)
    score = soft_min / hull_area
    return -score

def _local_search(pts, max_iter=500, tol=1e-10, alpha=20.0):
    """Local search using L-BFGS-B on the negative soft score."""
    x0 = pts.flatten()
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    def obj(x):
        return _neg_score_soft(x, alpha)
        
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score_hard(best_points)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Optimize regular polygon
    if time.time() < deadline - 60:
        try:
            refined = _local_search(best_points, max_iter=500, alpha=20.0)
            s = _score_hard(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass

    # Phase 2: Increase alpha for sharper minimum and refine
    if time.time() < deadline - 30:
        try:
            refined = _local_search(best_points, max_iter=500, alpha=50.0)
            s = _score_hard(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass

    # Phase 3: Perturbation restarts
    for i in range(15):
        if time.time() > deadline - 15:
            break
        
        noise_scale = 0.02
        perturbation = rng.normal(0, noise_scale, (N, 2))
        perturbed = best_points + perturbation
        
        try:
            # Use moderate alpha for restarts
            refined = _local_search(perturbed, max_iter=300, alpha=30.0)
            s = _score_hard(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            refined = _local_search(best_points, max_iter=1000, alpha=100.0, tol=1e-12)
            s = _score_hard(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass

    return best_points
