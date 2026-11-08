import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
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
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_objective(params, beta=100.0):
    """Smoothed objective for L-BFGS-B. Minimizes negative soft-min area / hull area."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    # Soft min approximation
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    if max_val == -np.inf:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    soft_min = -log_sum_exp / beta
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    return -soft_min / hull_area

def _refine_local(points, t0, deadline):
    """Refine a single point configuration using L-BFGS-B and Nelder-Mead."""
    best_points = points.copy()
    best_score = _exact_score(best_points)
    
    # Step 1: L-BFGS-B on soft objective
    # We use a higher beta for a sharper approximation of the min
    if time.time() - t0 < deadline - 5:
        try:
            x0 = best_points.flatten()
            # Use a moderate beta for smoothness
            res1 = minimize(_soft_objective, x0, args=(100.0,), method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            cand = res1.x.reshape(-1, 2)
            score1 = _exact_score(cand)
            if score1 > best_score:
                best_score = score1
                best_points = cand
        except Exception:
            pass

    # Step 2: Nelder-Mead on exact objective for final polish
    if time.time() - t0 < deadline - 2:
        try:
            x0 = best_points.flatten()
            res2 = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), x0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            cand = res2.x.reshape(-1, 2)
            score2 = _exact_score(cand)
            if score2 > best_score:
                best_score = score2
                best_points = cand
        except Exception:
            pass

    return best_points, best_score

def construct():
    n = N
    t0 = time.time()
    # Safety margin: The limit is 120s, we aim to finish by 110s
    deadline = t0 + 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: 
    # 1. Global search using Differential Evolution
    # 2. Local refinement of the best global candidate
    
    # Bounds for differential evolution
    # Points in [-1, 1] x [-1, 1] is a reasonable starting box
    bounds = [(-1, 1)] * (2 * n)
    
    # Objective function for DE
    def de_objective(x):
        points = x.reshape(-1, 2)
        return -_exact_score(points)
    
    # Run Differential Evolution
    # We give it about 80 seconds
    de_time_limit = 80.0
    try:
        # seed for reproducibility in DE if possible, though it's stochastic
        # maxiter and popsize control the search depth
        # We rely on the time limit to stop it early if needed
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=False  # We will do our own refinement
        )
        
        if res_de.success or res_de.fun < 0: # Check if it found something
            cand_points = res_de.x.reshape(-1, 2)
            cand_score = _exact_score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points
    except Exception:
        pass

    # Check time
    if best_points is None:
        # Fallback to regular polygon if DE failed or timed out without result
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _exact_score(best_points)

    # Refine the best candidate found
    if time.time() < deadline - 5:
        refined_pts, refined_score = _refine_local(best_points, t0, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts

    # Final check: if we still have time, maybe try a few random restarts of local refinement?
    # But DE usually covers this. Let's stick to the main path to ensure we return in time.
    
    return best_points
