# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_soft_bottleneck(x, beta, max_dist=1.0):
    """
    Objective function for the soft bottleneck optimization.
    We want to maximize d_min subject to d_i <= max_dist for all i.
    
    Minimize: -soft_min + lambda * penalty
    """
    pts = x.reshape(16, 2)
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return 0.0
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    neg_d = -beta * dists
    max_neg_d = np.max(neg_d)
    log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
    soft_min = -log_sum_exp_neg / beta
    
    # We want to MAXIMIZE soft_min, so we MINIMIZE -soft_min
    obj = -soft_min
    
    # Penalty for violating d_i <= 1.0
    exceed = np.maximum(0.0, dists - max_dist)
    penalty = np.sum(exceed ** 2)
    
    lambda_pen = 10.0
    return obj + lambda_pen * penalty

def _refine_exact(points):
    """
    Refine points using exact objective with L-BFGS-B.
    """
    def exact_obj(x):
        pts = x.reshape(16, 2)
        dists = spd.pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return -(d_min / d_max) ** 2
    
    res = minimize(exact_obj, points.flatten(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success or res.fun < exact_obj(points.flatten()):
        best_pts = res.x.reshape(16, 2)
        # Normalize
        best_pts = best_pts - best_pts.mean(axis=0)
        norm = np.max(np.linalg.norm(best_pts, axis=1))
        if norm > 1e-10:
            best_pts = best_pts / norm
        return best_pts
    return points

def _generate_hexagonal_16():
    """Generate a hexagonal lattice pattern for 16 points."""
    pts = []
    a = 1.0 # spacing
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = a * (q + r/2.0)
            y = a * (np.sqrt(3)/2.0 * r)
            pts.append([x, y])
    pts = np.array(pts)
    
    # We have 25 points. We need 16.
    dists_from_center = np.linalg.norm(pts, axis=1)
    sorted_indices = np.argsort(dists_from_center)
    
    selected = pts[sorted_indices[:16]]
    
    # Center and normalize
    selected = selected - selected.mean(axis=0)
    norm = np.max(np.linalg.norm(selected, axis=1))
    if norm > 1e-10:
        selected = selected / norm
    return selected

def _generate_random_16():
    """Generate random points in a unit square, then normalize."""
    pts = np.random.rand(16, 2) * 2 - 1 # [-1, 1]
    pts = pts - pts.mean(axis=0)
    norm = np.max(np.linalg.norm(pts, axis=1))
    if norm > 1e-10:
        pts = pts / norm
    return pts

def _nelder_mead_optimize(points, time_limit):
    """
    Use Nelder-Mead simplex method for optimization.
    This is a derivative-free method that can handle the non-convexity of the packing problem better than gradient-based methods in some basins.
    """
    start_time = time.time()
    
    # Objective: Minimize -R (exact score)
    # We add a small penalty for points going too far to keep the simplex bounded
    def obj(x):
        pts = x.reshape(16, 2)
        # Check time
        if time.time() > time_limit:
            return 1e9 # Large penalty to stop search
        
        # Regularization to keep points bounded
        reg = 0.001 * np.sum(pts**2)
        
        score = _score(pts)
        return -score + reg

    # Initial simplex size
    # Nelder-Mead requires a simplex of size N+1 in N dimensions.
    # We use a small perturbation for the other vertices.
    x0 = points.flatten()
    
    # Nelder-Mead options
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', options=options)
        
        if res.fun < obj(x0):
            best_pts = res.x.reshape(16, 2)
            # Normalize
            best_pts = best_pts - best_pts.mean(axis=0)
            norm = np.max(np.linalg.norm(best_pts, axis=1))
            if norm > 1e-10:
                best_pts = best_pts / norm
            return best_pts
    except Exception:
        pass
        
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initializations
    inits = []
    inits.append(_generate_hexagonal_16())
    
    # Add a few random inits
    for _ in range(5):
        inits.append(_generate_random_16())
        
    # Also try a few more structured inits if time permits
    # E.g. slightly perturbed hexagonal
    for i in range(3):
        hex_pts = _generate_hexagonal_16()
        noise = np.random.normal(0, 0.05, hex_pts.shape)
        perturbed = hex_pts + noise
        perturbed = perturbed - perturbed.mean(axis=0)
        norm = np.max(np.linalg.norm(perturbed, axis=1))
        if norm > 1e-10:
            perturbed = perturbed / norm
        inits.append(perturbed)

    # Optimization Strategy:
    # 1. Nelder-Mead (Global/Robust local)
    # 2. L-BFGS-B Polish (Precise local)
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        current_pts = init_pts.copy()
        
        # Phase 1: Nelder-Mead Optimization
        # This helps escape shallow local minima that gradient methods might get stuck in
        try:
            nm_pts = _nelder_mead_optimize(current_pts, deadline - 20)
            current_pts = nm_pts
        except Exception:
            pass
        
        # Phase 2: Exact Refinement with L-BFGS-B
        # Polish the result from Nelder-Mead
        try:
            refined_pts = _refine_exact(current_pts)
            current_pts = refined_pts
        except Exception:
            pass
        
        # Evaluate
        score = _score(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
        # If we have time, do a second round of Nelder-Mead from the polished point?
        # Or just move to next init.
        
    # If best_points is still None (shouldn't happen), fallback
    if best_points is None:
        best_points = _generate_hexagonal_16()
        
    # Final check and normalization
    best_points = best_points - best_points.mean(axis=0)
    norm = np.max(np.linalg.norm(best_points, axis=1))
    if norm > 1e-10:
        best_points = best_points / norm
        
    return best_points

# EVOLVE-BLOCK-END
