import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

# Precompute triangle indices
tri_indices = np.array(list(combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the ratio of minimum triangle area to convex hull area.
    """
    # Ensure points are float
    points = np.asarray(points, dtype=float)
    
    # Calculate areas of all triangles
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    
    # Cross product magnitude for area
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # If hull fails (e.g., collinear points), return 0 score
        return 0.0
        
    if hull_area == 0:
        return 0.0
        
    return min_area / hull_area

def neg_score(points):
    """
    Negative score for minimization.
    """
    return -compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initializations
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Grid-like
    init_grid = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]
    ])
    
    # 3. Random
    rng = np.random.default_rng(42)
    init_rand = rng.random((N, 2))
    
    initializations = [init_reg, init_grid, init_rand]
    
    for init in initializations:
        if time.time() > deadline:
            break
            
        # Run Nelder-Mead
        # x0 is flat array
        x0 = init.flatten()
        
        # Options for Nelder-Mead
        # maxiter: limit iterations to prevent hanging
        # xatol, fatol: tolerances
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon if optimization failed
        best_points = init_reg
    else:
        # Ensure the best points are the ones with the highest score
        # Re-evaluate to be sure
        # The loop above updates best_points only if score improves
        pass

    return best_points
