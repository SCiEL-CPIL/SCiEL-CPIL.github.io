import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def construct():
    start_time = time.time()
    deadline = 100.0 # 100 seconds safety margin
    
    # 1. Initialization: Fermat Spiral
    i = np.arange(N)
    r = np.sqrt(i / N)
    theta = i * 2.399963229728653 # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x0 = np.column_stack((x, y))
    x0 -= np.mean(x0, axis=0) # Center
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def objective(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        
        # Calculate areas of all triangles
        # Area = 0.5 * |det(P2-P1, P3-P1)|
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
                (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = np.abs(cross) * 0.5
        
        min_area = np.min(areas)
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            # If points are collinear or degenerate, return 0 (bad)
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
            
        return min_area / hull_area
    
    # We want to maximize objective, so minimize -objective
    def neg_objective(pts_flat):
        return -objective(pts_flat)
    
    # Optimization
    best_result = x0
    best_score = objective(x0)
    
    # Nelder-Mead options
    options = {
        'xatol': 1e-4,
        'fatol': 1e-6,
        'maxiter': 5000
    }
    
    try:
        result = minimize(neg_objective, x0.flatten(), method='Nelder-Mead', options=options)
        
        if result.success or result.fun < -best_score:
            best_result = result.x.reshape(-1, 2)
            best_score = -result.fun
            
        # Check time
        if time.time() - start_time > deadline:
            pass # Just return best found so far
            
        # Optional: Second pass with tighter tolerance if time remains
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time > 20:
             options2 = {
                'xatol': 1e-6,
                'fatol': 1e-8,
                'maxiter': 5000
             }
             result2 = minimize(neg_objective, best_result.flatten(), method='Nelder-Mead', options=options2)
             if result2.fun < -best_score:
                 best_result = result2.x.reshape(-1, 2)
                 best_score = -result2.fun
                 
    except Exception as e:
        # Fallback to initial if optimization fails
        pass
        
    return best_result
