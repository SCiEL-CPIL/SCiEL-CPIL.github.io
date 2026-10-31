import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13

def _compute_triangle_areas(pts):
    """
    Vectorized computation of areas for all triples of points.
    """
    n = len(pts)
    # Precompute all combinations of 3 indices
    if not hasattr(_compute_triangle_areas, 'idx'):
        _compute_triangle_areas.idx = np.array(list(combinations(range(n), 3)))
    idx = _compute_triangle_areas.idx
    
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _get_hull_area(pts):
    """
    Compute the area of the convex hull of the points.
    """
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback if points are collinear or identical
        return 0.0

def _score(pts, beta=100.0):
    """
    Compute the score: min triangle area / hull area.
    Uses soft-min for differentiability if needed, but here we use hard min for final eval.
    For optimization, we might use soft min.
    """
    hull_area = _get_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score_soft(params, beta):
    """
    Objective function for optimization using soft-min.
    """
    pts = params.reshape(N, 2)
    hull_area = _get_hull_area(pts)
    if hull_area < 1e-12:
        return 1e6  # Large penalty for degenerate hull
    
    areas = _compute_triangle_areas(pts)
    
    # Soft min
    # To avoid overflow/underflow, shift by min area
    a_min = np.min(areas)
    # If beta is very large, this becomes unstable. 
    # soft_min = a_min - (1/beta) * log(sum(exp(-beta*(areas - a_min))))
    # Since areas - a_min >= 0, exp(-beta*(areas - a_min)) <= 1.
    # Sum is between 1 and 286. Log is safe.
    shifted = -beta * (areas - a_min)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = a_min - log_sum / beta
    
    return -soft_min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_points = x0.reshape(N, 2).copy()
    best_score = _score(best_points)
    
    # 1. Global Search with Differential Evolution
    # Use a moderate beta for soft-min to guide the DE
    beta_de = 50.0
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    try:
        # DE is slow, so limit iterations
        result_de = differential_evolution(
            lambda x: _neg_score_soft(x, beta_de),
            bounds,
            seed=42,
            maxiter=30,
            tol=1e-4,
            polish=False,
            init='sobol',
            popsize=10
        )
        
        candidate = result_de.x.reshape(N, 2)
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_points = candidate.copy()
            
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_points

    # 2. Local Refinement
    # We use a sequence of minimizations with increasing beta (sharper min)
    # Start from the best point found so far
    current_points = best_points.flatten()
    
    betas = [50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better than L-BFGS-B for this specific objective
            # Or L-BFGS-B with soft min
            result_lm = minimize(
                lambda x: _neg_score_soft(x, beta),
                current_points,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            candidate = result_lm.x.reshape(N, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()
                current_points = candidate.flatten()
            else:
                # Even if score didn't improve in hard metric, the soft metric might have moved us closer to a better local min
                # Update current points to continue the chain
                current_points = result_lm.x
                
        except Exception:
            pass
            
    # Final check
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
        
    return best_points
