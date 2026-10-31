import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    # Generate indices for all unique triples i < j < k
    # Using np.triu_indices or manual loops?
    # For N=13, manual loops in Python are ~286 iterations.
    # Vectorized:
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i < j) & (j < k)
    i_idx = i[mask]
    j_idx = j[mask]
    k_idx = k[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def compute_hull_area(points):
    try:
        # Check for degenerate cases
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    # Scale to unit area
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = points * scale
    min_tri_area = compute_min_triangle_area(pts_scaled)
    return min_tri_area

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def objective(x):
        points = x.reshape(N, 2)
        # Center to remove translation symmetry
        points = points - np.mean(points, axis=0)
        
        hull_area = compute_hull_area(points)
        if hull_area < 1e-10:
            return 1e6
            
        # Normalize to unit hull area to remove scale symmetry
        scale = 1.0 / np.sqrt(hull_area)
        pts_scaled = points * scale
        
        min_tri = compute_min_triangle_area(pts_scaled)
        
        # Maximize min_tri -> Minimize -min_tri
        return -min_tri

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        # Global Search: Differential Evolution
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-8,
            seed=42,
            polish=False
        )
        
        best_points = res.x.reshape(N, 2)
        
        # Local Refinement: Nelder-Mead
        if time.time() - start_time < time_limit - 5.0:
            res_local = minimize(
                objective, 
                res.x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_local.fun < res.fun:
                best_points = res_local.x.reshape(N, 2)
                
        return best_points

    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
