"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points."""
    n = len(points)
    min_area = float('inf')
    # Iterate over all triples
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the final score: min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _softmin_objective(points_flat, beta=10.0):
    """
    Objective function for minimization.
    We want to maximize min_area / hull_area.
    So we minimize - (softmin(areas) / hull_area).
    """
    points = points_flat.reshape(-1, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    
    n = len(points)
    # Calculate all triangle areas
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    # Soft minimum: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max area? No, we want min, so we want large negative exp for small areas.
    # Standard softmin: min(x) ~ -1/beta log(sum(exp(-beta x)))
    # If x is small, exp(-beta x) is large.
    # Let's shift to prevent overflow: let x_min = min(areas)
    # sum(exp(-beta (x - x_min))) * exp(-beta x_min)
    # log(sum(exp(-beta x))) = log(sum(exp(-beta (x - x_min)))) - beta x_min
    # softmin = -1/beta * (log_sum_exp_term - beta x_min) = x_min - 1/beta log_sum_exp_term
    # This is more stable.
    x_min = np.min(areas)
    shifted = areas - x_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = x_min - (1.0 / beta) * log_sum
    
    objective_val = - (soft_min_val / hull_area)
    return objective_val

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Initial configurations
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Regular 12-gon + center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    init2 = np.vstack([poly12, [[0, 0]]])
    
    # 3. Random points on circle
    rng = np.random.RandomState(42)
    angles_rand = rng.uniform(0, 2 * np.pi, N)
    radii_rand = rng.uniform(0.5, 1.0, N)
    init3 = np.column_stack([radii_rand * np.cos(angles_rand), radii_rand * np.sin(angles_rand)])
    
    inits = [init1, init2, init3]
    
    best_points = init1
    best_score = _score(init1)
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        x0 = init.flatten()
        
        # Try L-BFGS-B with different betas (annealing)
        betas = [5.0, 10.0, 20.0, 50.0]
        
        current_x = x0
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    _softmin_objective, 
                    current_x, 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                current_x = res.x
            except Exception:
                pass
        
        final_points = current_x.reshape(-1, 2)
        final_score = _score(final_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
            
    return best_points
# EVOLVE-BLOCK-END
