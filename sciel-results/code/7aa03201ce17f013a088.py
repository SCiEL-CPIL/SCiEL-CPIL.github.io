import numpy as np
import time
from scipy.optimize import minimize

def _compute_polygon_score(points):
    """
    Computes the score for a given set of points (20, 2).
    Returns the minimum vertex score.
    """
    n = len(points)
    # Normalize
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    dists_to_centroid = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points_norm = points_norm / mean_dist

    # Check strictly convex
    # 1. No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points_norm[i] - points_norm[j]) < 1e-9:
                return 0.0
            
    # 2. Cyclic order and convexity
    signs = []
    for i in range(n):
        p1 = points_norm[i]
        p2 = points_norm[(i+1) % n]
        p3 = points_norm[(i+2) % n]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            return 0.0
        signs.append(np.sign(cross))
        
    if not all(s == signs[0] for s in signs):
        return 0.0

    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = points_norm[i]
        diffs = points_norm - v
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        dists_sorted = np.sort(dists)
        
        if np.any(dists_sorted < 0.05):
            min_score = 0.0
            continue
            
        gaps = dists_sorted[1:] - dists_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (dists_sorted[k] + dists_sorted[k+1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            continue
            
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17:
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = dists_sorted[idxs]
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        
        if raw == 0:
            score_i = 1.0
        else:
            score_i = min(1.0, 1.0 / raw)
            
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def _convexity_penalty(points):
    """Compute penalty for non-convexity. Returns 0 if strictly convex."""
    n = len(points)
    penalty = 0.0
    for i in range(n):
        p1 = points[i]
        p2 = points[(i+1) % n]
        p3 = points[(i+2) % n]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < 0:
            penalty += cross**2
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 0.01:
                penalty += (0.01 - d)**2
    return penalty

def _objective_flat(flat_points):
    """Objective for flat array of 40 coordinates (20x2)."""
    points = flat_points.reshape(20, 2)
    score = _compute_polygon_score(points)
    if score == 0.0:
        # Add convexity penalty to guide optimization
        pen = _convexity_penalty(points)
        return 1.0 + pen
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = 630.0
    margin = 30.0
    
    # Initial guess: Regular polygon
    angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0 = np.cos(angles0)
    y0 = np.sin(angles0)
    points0 = np.column_stack((x0, y0))
    best_score = _compute_polygon_score(points0)
    best_points = points0.copy()
    
    def time_remaining():
        return deadline - time.time() - margin
    
    # Stage 1: Nelder-Mead on flat coordinates
    if time_remaining() > 60:
        try:
            flat0 = points0.flatten()
            res = minimize(_objective_flat, flat0, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < -best_score:
                points_opt = res.x.reshape(20, 2)
                score_opt = _compute_polygon_score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B refinement
    if time_remaining() > 60:
        try:
            flat_best = best_points.flatten()
            res2 = minimize(_objective_flat, flat_best, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < -best_score:
                points_opt2 = res2.x.reshape(20, 2)
                score_opt2 = _compute_polygon_score(points_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_points = points_opt2.copy()
        except Exception:
            pass
    
    # Stage 3: Final Nelder-Mead polish with tighter tolerances
    if time_remaining() > 30:
        try:
            flat_best = best_points.flatten()
            res3 = minimize(_objective_flat, flat_best, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res3.fun < -best_score:
                points_opt3 = res3.x.reshape(20, 2)
                score_opt3 = _compute_polygon_score(points_opt3)
                if score_opt3 > best_score:
                    best_score = score_opt3
                    best_points = points_opt3.copy()
        except Exception:
            pass
    
    # Stage 4: Try a few random perturbations of the best solution
    if time_remaining() > 30:
        rng = np.random.RandomState(42)
        for attempt in range(5):
            if time_remaining() < 10:
                break
            try:
                perturbation = rng.normal(0, 0.01, size=(20, 2))
                perturbed = best_points + perturbation
                flat_pert = perturbed.flatten()
                res4 = minimize(_objective_flat, flat_pert, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res4.fun < -best_score:
                    points_opt4 = res4.x.reshape(20, 2)
                    score_opt4 = _compute_polygon_score(points_opt4)
                    if score_opt4 > best_score:
                        best_score = score_opt4
                        best_points = points_opt4.copy()
            except Exception:
                pass
    
    return best_points
