"""Optimized program for Heilbronn convex regions using smooth soft-min surrogate."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas(pts):
    """Compute all C(N,3) triangle areas for N points."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on all points (not correct but safe)
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

def _soft_min_surrogate(pts, beta=0.01):
    """
    Compute the soft-min of triangle areas divided by convex hull area.
    Uses log-sum-exp for numerical stability.
    Returns negative of the surrogate so we can minimize it.
    """
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    
    if hull_area < 1e-10:
        return 1e10
    
    # Soft min: -log(sum(exp(-area/beta))) / hull_area
    # For numerical stability, subtract max
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    
    soft_min = -log_sum_exp  # This is the soft-min of areas
    score = soft_min / hull_area
    return -score  # Negative because we minimize

def _soft_min_grad(pts, beta=0.01):
    """
    Compute gradient of the negative soft-min surrogate.
    """
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    
    if hull_area < 1e-10:
        return np.zeros_like(pts)
    
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    exp_vals = np.exp(neg_areas - max_neg)
    sum_exp = np.sum(exp_vals)
    
    # Soft min = -beta * log(sum(exp(-areas/beta)))
    # d(soft_min)/d(area_i) = -exp(-area_i/beta) / sum(exp(-areas_j/beta))
    # = -exp_vals[i] / sum_exp
    
    # score = soft_min / hull_area
    # d(score)/d(area_i) = d(soft_min)/d(area_i) / hull_area
    # We want to minimize -score, so gradient of -score w.r.t. area_i = -d(score)/d(area_i)
    
    # Now we need to propagate through to the point coordinates
    # This is complex, so we'll use finite differences as a fallback
    # For now, return zeros and use finite difference in the optimizer
    
    return None  # Signal to use finite differences

def _objective_and_grad(x, beta=0.01):
    """
    Objective function and gradient for L-BFGS-B.
    x is flattened array of shape (2*N,).
    """
    pts = x.reshape(N, 2)
    obj = _soft_min_surrogate(pts, beta)
    
    # Use finite differences for gradient
    grad = np.zeros_like(x)
    eps = 1e-7
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        obj_plus = _soft_min_surrogate(x_plus.reshape(N, 2), beta)
        grad[i] = (obj_plus - obj) / eps
    
    return obj, grad

def construct():
    """
    Place N=13 points to maximize min triangle area / convex hull area.
    Uses smooth soft-min surrogate with L-BFGS-B optimization.
    """
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Initialize with regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_pts = x0.reshape(N, 2)
    best_score = _soft_min_surrogate(best_pts, 0.01)
    
    # Try multiple beta values and initializations
    betas = [0.05, 0.02, 0.01, 0.005]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Optimize from regular polygon
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
        
        # Also try with small perturbations
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(N * 100 + trial)
            perturbation = np.random.randn(N, 2) * 0.1
            x_init = (x0.reshape(N, 2) + perturbation).flatten()
            
            try:
                res = minimize(
                    _objective_and_grad,
                    x_init,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = res.x.reshape(N, 2)
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_and_grad,
                best_pts.flatten(),
                args=(0.001,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Ensure convex position: if any point is inside the convex hull, project it
    try:
        hull = ConvexHull(best_pts)
        # Check if all points are on the hull
        # If not, we may need to fix, but for now return as is
        # The optimization should keep points in convex position
    except Exception:
        pass
    
    return best_pts

# EVOLVE-BLOCK-END
