"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(N), 3):
        area = _triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    return min_area / hull_area

def _objective(x):
    points = x.reshape(-1, 2)
    return -_score(points)

def construct():
    # Initial guess: 4-4-4-1 grid
    coords = [-1.5, -0.5, 0.5, 1.5]
    init_points = []
    for x in coords:
        init_points.append([x, -1.5])
    for x in coords:
        init_points.append([x, -0.5])
    for x in coords:
        init_points.append([x, 0.5])
    init_points.append([0.0, 1.5])
    
    init_x = np.array(init_points).flatten()
    
    best_points = np.array(init_points)
    best_score = _score(best_points)
    
    # Try a few different initial configurations to avoid local minima
    # Config 2: 4x4 grid minus 3 corners (asymmetric but different topology)
    # Config 3: Random perturbations of the grid
    
    # Run Nelder-Mead
    try:
        res = minimize(_objective, init_x, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-8})
        candidate_points = res.x.reshape(-1, 2)
        candidate_score = _score(candidate_points)
        if candidate_score > best_score:
            best_points = candidate_points
            best_score = candidate_score
    except Exception:
        pass
        
    return best_points
# EVOLVE-BLOCK-END
