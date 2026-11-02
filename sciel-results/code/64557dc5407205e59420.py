import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # Points shape: (N, 2)
    # We can generate all pairs of indices
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    x1, y1 = p_i[:, 0], p_i[:, 1]
    x2, y2 = p_j[:, 0], p_j[:, 1]
    x3, y3 = p_k[:, 0], p_k[:, 1]
    
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    if len(areas) > 0:
        return np.min(areas)
    return 0.0

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback if points are collinear or degenerate
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    min_tri_area = _min_triangle_area(points)
    return min_tri_area / hull_area

def _objective(x):
    """Objective function for minimization. We want to maximize score, so minimize negative score."""
    points = x.reshape(-1, 2)
    # Add a small penalty if points are too close to avoid numerical issues?
    # The score function handles division by zero.
    return -_score(points)

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Regular 14-gon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0
    init_regular = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # 2. Random points in a circle (multiple starts)
    rng = np.random.default_rng(42)
    
    configs = [init_regular]
    
    # Add a few random perturbations of the regular polygon
    for _ in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        configs.append(init_regular + noise)
        
    # Add a few purely random configurations in a circle
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        configs.append(np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))

    for config in configs:
        if time.time() - start_time > time_limit:
            break
            
        # Use Nelder-Mead for non-smooth optimization
        # maxiter needs to be reasonable.
        # N=14 -> 28 variables.
        try:
            res = minimize(
                _objective, 
                config.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.success or res.fun < -best_score:
                candidate_points = res.x.reshape(-1, 2)
                score = _score(candidate_points)
                if score > best_score:
                    best_score = score
                    best_points = candidate_points
        except Exception:
            pass

    # If we have time, do a final polish with tighter tolerances on the best point
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            res_final = minimize(
                _objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate_points = res_final.x.reshape(-1, 2)
            score = _score(candidate_points)
            if score > best_score:
                best_points = candidate_points
                best_score = score
        except Exception:
            pass

    # If no valid points found, return the initial regular polygon
    if best_points is None:
        best_points = init_regular

    return best_points

# EVOLVE-BLOCK-END
