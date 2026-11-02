"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.special import logsumexp
import itertools

def _triangle_areas(pts):
    """Compute all triangle areas for N points using vectorized cross products."""
    n = len(pts)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    pi = pts[i_flat]
    pj = pts[j_flat]
    pk = pts[k_flat]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_triangle_area(pts, temperature=0.01):
    """
    Compute a smooth surrogate for the minimum triangle area.
    """
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    neg_areas = -areas / temperature
    log_sum = logsumexp(neg_areas)
    softmin = -temperature * log_sum
    return softmin

def _neg_soft_score(pts, temperature=0.01):
    """
    Negative soft score for optimization.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    softmin = _soft_min_triangle_area(pts, temperature)
    return -softmin / hull_area

def _neg_exact_score(pts_flat):
    """
    Negative exact score for final polish.
    """
    pts = pts_flat.reshape(-1, 2)
    return -_score_points(pts)

def _generate_7_6_config(r_ratio, phase):
    """
    Generate 7 outer points (radius 1) and 6 inner points (radius r_ratio).
    Outer angles: 2*pi*k/7
    Inner angles: 2*pi*k/7 + phase
    """
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + phase
    inner_pts = r_ratio * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _neg_symmetric_score_7_6(params):
    r_ratio, phase = params
    if r_ratio < 0 or r_ratio > 1.5: # Allow some flexibility
        return 1e10
    pts = _generate_7_6_config(r_ratio, phase)
    return _neg_soft_score(pts, temperature=0.01)

def _generate_1_6_6_config(r1, r2, phase):
    """
    Generate 1 center, 6 inner (radius r1), 6 outer (radius r2).
    Inner angles: 2*pi*k/6
    Outer angles: 2*pi*k/6 + phase
    """
    center = np.array([[0.0, 0.0]])
    
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_pts = r1 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + phase
    outer_pts = r2 * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    pts = np.vstack([center, inner_pts, outer_pts])
    return pts

def _neg_symmetric_score_1_6_6(params):
    r1, r2, phase = params
    if r1 < 0 or r2 < 0 or r2 < r1: # Outer should be larger
        return 1e10
    pts = _generate_1_6_6_config(r1, r2, phase)
    return _neg_soft_score(pts, temperature=0.01)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pts = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_score
        try:
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # 1. Symmetric Optimization Phase
    
    # Strategy A: 7 Outer, 6 Inner
    # Initial guesses for (r_ratio, phase)
    initial_guesses_A = [
        (0.5, np.pi/7),
        (0.6, np.pi/7),
        (0.7, np.pi/7),
        (0.8, 0.0),
        (0.5, 0.0),
        (0.6, 0.0),
    ]
    
    for guess in initial_guesses_A:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(_neg_symmetric_score_7_6, guess, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
            if res.fun < 1e9:
                r_ratio, phase = res.x
                pts = _generate_7_6_config(r_ratio, phase)
                update_best(pts)
        except Exception:
            pass
            
    # Strategy B: 1 Center, 6 Inner, 6 Outer
    initial_guesses_B = [
        (0.5, 1.0, 0.0),
        (0.5, 1.0, np.pi/6),
        (0.6, 1.0, np.pi/6),
        (0.4, 0.9, np.pi/6),
    ]
    
    for guess in initial_guesses_B:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(_neg_symmetric_score_1_6_6, guess, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
            if res.fun < 1e9:
                r1, r2, phase = res.x
                pts = _generate_1_6_6_config(r1, r2, phase)
                update_best(pts)
        except Exception:
            pass

    # 2. Final Polish Phase
    # If we have a good candidate, refine it using a general local search
    if best_pts is not None:
        x0 = best_pts.flatten()
        
        # Use Powell for derivative-free refinement on all 26 coordinates
        # We use a tight tolerance to ensure we don't drift too far from the good solution
        try:
            res = minimize(
                _neg_exact_score,
                x0,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 0: # Valid score
                refined_pts = res.x.reshape(-1, 2)
                update_best(refined_pts)
                
                # One more polish with even tighter tolerance if time permits
                if time.time() < deadline - 5.0:
                    res2 = minimize(
                        _neg_exact_score,
                        refined_pts.flatten(),
                        method='Powell',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    if res2.fun < 0:
                        final_pts = res2.x.reshape(-1, 2)
                        update_best(final_pts)
        except Exception:
            pass

    # Fallback if symmetric search failed (should not happen)
    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.rand(N, 2)
        update_best(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
