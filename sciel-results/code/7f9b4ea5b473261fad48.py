"""Heilbronn convex regions - optimized placement of 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of convex hull of points using scipy."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area if all points are in convex position
        n = len(points)
        if n < 3:
            return 1e-10
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) * 0.5

def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _regular_polygon(n, center=None, radius=1.0, phase=0.0):
    """Generate vertices of a regular n-gon."""
    if center is None:
        center = np.array([0.0, 0.0])
    angles = phase + 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area.
    We minimize - (soft_min(areas) / hull_area).
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min: -logsumexp(-beta * areas) / beta
    # To avoid overflow in exp, we subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    # If max_val is very large negative, exp will underflow to 0, sum will be 0, log(0) is -inf.
    # But areas are positive, so neg_beta_areas are negative.
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def _two_shell_points(r1, r2, p1, p2, n1):
    """Generate points in two concentric shells."""
    n1 = int(round(n1))
    n1 = max(1, min(N - 1, n1))
    n2 = N - n1
    pts1 = _regular_polygon(n1, r1, p1)
    pts2 = _regular_polygon(n2, r2, p2)
    return np.vstack([pts1, pts2])

def construct():
    """Place 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill
    
    # Start with regular polygon
    points = _regular_polygon(N)
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Phase 1: Try various starting configurations
    starts = [points.copy()]
    
    # Two-shell configurations with various splits
    for n1 in range(3, N - 2):
        n2 = N - n1
        for r1 in [0.3, 0.5, 0.7]:
            for r2 in [0.9, 1.0, 1.2]:
                for p1 in [0.0, np.pi / n1]:
                    for p2 in [0.0, np.pi / n2]:
                        candidate = _two_shell_points(r1, r2, p1, p2, n1)
                        cand_score = _score(candidate)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_points = candidate.copy()
    
    # Phase 2: L-BFGS-B optimization with soft-min surrogate
    if time.time() - start_time < time_limit - 20:
        try:
            x0 = best_points.flatten()
            res1 = minimize(_soft_min_objective, x0, args=(50.0,), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if time.time() - start_time > time_limit - 20:
                return best_points
                
            candidate = res1.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Powell method for further refinement
    if time.time() - start_time < time_limit - 15:
        try:
            x0 = best_points.flatten()
            res2 = minimize(_soft_min_objective, x0, args=(100.0,), method='Powell',
                            options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if time.time() - start_time > time_limit - 15:
                return best_points
                
            candidate = res2.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish with high beta
    if time.time() - start_time < time_limit - 10:
        try:
            x0 = best_points.flatten()
            res3 = minimize(_soft_min_objective, x0, args=(200.0,), method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() - start_time > time_limit - 10:
                return best_points
                
            candidate = res3.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 5: Fine-tuning with very high beta
    if time.time() - start_time < time_limit - 5:
        try:
            x0 = best_points.flatten()
            res4 = minimize(_soft_min_objective, x0, args=(500.0,), method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11})
            
            if time.time() - start_time > time_limit - 5:
                return best_points
                
            candidate = res4.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points
