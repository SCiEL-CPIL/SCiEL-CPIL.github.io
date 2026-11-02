"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(pts[i], pts[j], pts[k])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is the area
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_triangle_area(pts, beta=50.0):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(pts[i], pts[j], pts[k]))
    areas = np.array(areas)
    # Use log-sum-exp for numerical stability
    m = np.min(areas)
    if m <= 0:
        return 0.0
    # Soft minimum approximation
    log_sum = m + np.log(np.sum(np.exp((areas - m) / beta)))
    return log_sum * beta / len(areas)  # Approximation of geometric mean scaled

def _neg_soft_score(pts, beta=50.0):
    """Negative of the soft score for minimization."""
    hull_area = _hull_area(pts)
    if hull_area <= 0:
        return 0.0
    soft_min = _soft_min_triangle_area(pts, beta)
    return -soft_min / hull_area

def _neg_score(pts):
    """Negative of the actual score for minimization."""
    return -_score_points(pts)

def _refine(pts, max_iter=1000, tol=1e-10):
    """Refine the point configuration using local optimization."""
    x0 = pts.flatten()
    n = len(pts)
    
    def obj(x):
        pts_x = x.reshape(n, 2)
        return _neg_soft_score(pts_x, beta=50.0)
    
    def jac(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
        return grad
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', jac=jac, 
                         options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        return result.x.reshape(n, 2)
    except Exception:
        return pts

def construct():
    """Construct 13 points to maximize the minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Initial configurations
    initial_configs = []
    
    # Regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(regular_poly)
    
    # Ring with some perturbation
    rng = np.random.RandomState(42)
    for trial in range(5):
        r = 1.0 + rng.normal(0, 0.1, N)
        r = np.clip(r, 0.5, 1.5)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.05, N)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        initial_configs.append(pts)
    
    # Random points in unit disk
    for trial in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_configs.append(pts)
    
    # Evaluate and refine each initial configuration
    for idx, pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Refine
        refined_pts = _refine(pts, max_iter=500, tol=1e-9)
        refined_score = _score_points(refined_pts)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
        
        # Additional refinement with tighter tolerance
        if time.time() < deadline - 20.0:
            further_refined = _refine(refined_pts, max_iter=1000, tol=1e-11)
            further_score = _score_points(further_refined)
            if further_score > best_score:
                best_score = further_score
                best_pts = further_refined.copy()
    
    # Final polish if time permits
    if best_pts is not None and time.time() < deadline - 10.0:
        final_refined = _refine(best_pts, max_iter=2000, tol=1e-12)
        final_score = _score_points(final_refined)
        if final_score > best_score:
            best_pts = final_refined
            best_score = final_score
    
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

# EVOLVE-BLOCK-END
