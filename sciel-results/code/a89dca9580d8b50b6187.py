import numpy as np
from scipy.optimize import minimize
import time

N = 11

def min_triangle_area_fast(points):
    n = len(points)
    if n < 3:
        return 0.0
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds buffer
    
    # Barycentric to Cartesian
    def bary_to_cart(bary):
        u = bary[:, 0]
        v = bary[:, 1]
        x = 0.5 + 0.5 * v - 0.5 * u
        y = (np.sqrt(3) / 2) * (1 - u - v)
        return np.column_stack((x, y))

    # Transform from [0,1]^2 per point to valid barycentric coords
    def transform_to_barycentric(transformed):
        bary = np.zeros((N, 2))
        u_prime = transformed[0::2]
        v_prime = transformed[1::2]
        bary[:, 0] = u_prime
        bary[:, 1] = v_prime * (1 - u_prime)
        return bary

    def objective_transformed(transformed_flat):
        bary_coords = transform_to_barycentric(transformed_flat)
        points = bary_to_cart(bary_coords)
        area = min_triangle_area_fast(points)
        return -area

    bounds = [(0, 1)] * (2 * N)
    
    best_val = -float('inf')
    best_x = None
    
    np.random.seed(42)
    
    # Multi-start
    num_starts = 500 # Increased starts due to faster objective
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        x0 = np.random.rand(2 * N)
        
        try:
            res = minimize(objective_transformed, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 100, 'maxfev': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
            
    if best_x is None:
        # Fallback
        bary_coords = np.zeros((N, 2))
        bary_coords[0] = [1, 0]
        bary_coords[1] = [0, 1]
        bary_coords[2] = [0, 0]
        bary_coords[3] = [1/3, 1/3]
        for j in range(4, N):
            bary_coords[j] = np.random.rand(2)
            # Ensure valid barycentric? The transform handles it, but here we are in bary space directly for fallback?
            # No, fallback returns points. We need to generate valid points.
            # Let's just use random points in the triangle for fallback.
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - v, 1 - u
            bary_coords[j] = [u, v]
        points = bary_to_cart(bary_coords)
        return points

    bary_coords = transform_to_barycentric(best_x)
    points = bary_to_cart(bary_coords)
    return points
