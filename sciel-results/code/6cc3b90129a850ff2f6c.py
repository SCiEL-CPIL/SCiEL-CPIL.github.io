import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 13

# Precompute indices for all triples to speed up score calculation
TRIPLES = np.array([(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)])

# EVOLVE-BLOCK-START

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
    """
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def objective(x):
    """
    Objective function for minimization (negative score).
    """
    points = x.reshape(N, 2)
    return -compute_score(points)

def generate_regular_polygon():
    """
    Generate a regular 13-gon centered at origin.
    """
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    radius = 0.5
    points = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in theta])
    return points

def local_search_nelder_mead(points, time_limit=10.0):
    """
    Refine points using Nelder-Mead optimization.
    """
    start_time = time.time()
    x_init = points.flatten()
    
    # Use Nelder-Mead with careful tolerances
    try:
        res = minimize(
            objective,
            x_init,
            method='Nelder-Mead',
            options={
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxiter': 5000,
                'maxfev': 5000 * 28,  # N=13, so 26 params, ~28 evaluations per iter
                'adaptive': True,
                'disp': False
            }
        )
        
        if time.time() - start_time > time_limit:
            # If we hit the time limit, use the current best
            pass
        
        score = -res.fun
        if score > 0 and not np.isnan(score):
            return res.x.reshape(N, 2), score
            
    except Exception:
        pass
    
    return points, compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, keep 10s buffer)
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Start with regular polygon and refine with Nelder-Mead
    # This is known to be a strong starting point for this problem
    base_points = generate_regular_polygon()
    
    # Multiple restarts with slight perturbations to escape local minima
    num_restarts = 8
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time for this restart
        remaining_restarts = num_restarts - i
        time_per_restart = max(5.0, (deadline - time.time() - 5.0) / remaining_restarts)
        
        # Perturb the regular polygon slightly for diversity
        if i == 0:
            init_points = base_points.copy()
        else:
            # Add small random perturbation
            noise = np.random.normal(0, 0.02 * (i + 1), (N, 2))
            init_points = base_points + noise
        
        # Run Nelder-Mead refinement
        pts, score = local_search_nelder_mead(init_points, time_limit=time_per_restart)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # If we've found a very good solution, we can break early
        if best_score > 42.0:
            break
    
    # If we still have time, do a final refinement on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            final_pts, final_score = local_search_nelder_mead(best_points, time_limit=remaining_time)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = base_points
        
    return best_points
# EVOLVE-BLOCK-END
