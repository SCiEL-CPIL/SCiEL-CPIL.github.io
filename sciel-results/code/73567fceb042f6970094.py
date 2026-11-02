import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
# Vertices of the equilateral triangle
# A0 = (0,0), A1 = (1,0), A2 = (0.5, sqrt(3)/2)
# Area of this triangle is sqrt(3)/4

def params_to_points(params):
    """
    Convert parameters (u, w) for each point to Cartesian coordinates.
    u, w in [0, 1].
    v = w * (1 - u)
    x = u + 0.5 * v
    y = sqrt(3)/2 * v
    """
    params = np.asarray(params, dtype=float)
    # Handle potential 0-d or wrong shape inputs gracefully
    if params.ndim == 0:
        # If scalar, we can't form 11 points. Return a default valid config.
        # Default: vertices and centroid
        h = np.sqrt(3) / 2
        return np.array([
            [0, 0], [1, 0], [0.5, h],
            [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
            [0.5, h/6], [0.5, 5*h/6], [0.5, h/2],
            [1/6, h/6], [5/6, h/6]
        ])
    if params.size != 2 * N:
        # If size is wrong, pad or trim? Better to return default to avoid crash.
        h = np.sqrt(3) / 2
        return np.array([
            [0, 0], [1, 0], [0.5, h],
            [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
            [0.5, h/6], [0.5, 5*h/6], [0.5, h/2],
            [1/6, h/6], [5/6, h/6]
        ])
        
    u = params[0::2]
    w = params[1::2]
    
    v = w * (1.0 - u)
    
    x = u + 0.5 * v
    y = (3.0 ** 0.5) / 2.0 * v
    
    points = np.column_stack((x, y))
    return points

def calculate_min_area(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of areas
    # We need to compute area for all i < j < k
    # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    
    # This can be memory intensive if N is large, but N=11 is small.
    # Number of triangles = 165.
    # We can just loop or use combinations.
    
    min_area = float('inf')
    
    # For efficiency, precompute differences?
    # Or just use itertools.combinations
    from itertools import combinations
    
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Vector cross product in 2D
        # det(p2-p1, p3-p1) = (p2x-p1x)*(p3y-p1y) - (p2y-p1y)*(p3x-p1x)
        det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * abs(det)
        
        if area < min_area:
            min_area = area
            
    return min_area

def objective_de(params):
    """
    Objective for Differential Evolution (minimization).
    Returns negative min area.
    """
    points = params_to_points(params)
    min_area = calculate_min_area(points)
    return -min_area

def objective_lbgfs(params, beta=50.0):
    """
    Objective for L-BFGS-B using soft-min approximation.
    Returns negative soft-min of areas.
    """
    points = params_to_points(params)
    n = len(points)
    
    # Calculate all areas
    areas = []
    from itertools import combinations
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * abs(det)
        areas.append(area)
        
    areas = np.array(areas)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, subtract max area?
    # log-sum-exp trick:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -beta * areas.
    # max(x) = -beta * min(areas).
    
    # We want soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # Let y = -beta * areas.
    # log_sum_exp(y) = max(y) + log(sum(exp(y - max(y))))
    # soft_min = -1/beta * (max(y) + log(sum(exp(y - max(y)))))
    
    y = -beta * areas
    max_y = np.max(y)
    log_sum_exp = max_y + np.log(np.sum(np.exp(y - max_y)))
    soft_min = -1.0 / beta * log_sum_exp
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Bounds for u and w: [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # 1. Differential Evolution
    best_params = None
    best_score = -float('inf')
    
    try:
        # DE is stochastic. Use a fixed seed for reproducibility in evaluation?
        # The task says "random" is allowed, but deterministic is often safer for "seed" comparison.
        # However, DE with random init is fine.
        # popsize=20, maxiter=100
        # tolerance=1e-8
        
        res = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            polish=False # We will polish manually
        )
        
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = -res.fun
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        if best_params is None:
            # Fallback to valid random points inside the triangle
            u = np.random.rand(N)
            w = np.random.rand(N)
            v = w * (1 - u)
            x = u + 0.5 * v
            y = (3.0 ** 0.5) / 2.0 * v
            return np.column_stack((x, y))
        else:
            return params_to_points(best_params)
            
    # 2. Local Refinement with L-BFGS-B
    if best_params is not None:
        # Ensure best_params is a 1D array of correct size
        best_params = np.asarray(best_params, dtype=float).flatten()
        if best_params.size != 2 * N:
            # If size mismatch, skip refinement
            return params_to_points(best_params)

        # Try a few betas for soft-min
        current_params = best_params
        current_val = -best_score # This is the min area
        
        for beta in [20, 50, 100]:
            try:
                res_lbfgs = minimize(
                    objective_lbgfs, 
                    current_params, 
                    args=(beta,),
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10}
                )
                
                # Check if res_lbfgs.x is valid
                if hasattr(res_lbfgs, 'x') and res_lbfgs.x is not None:
                    res_params = np.asarray(res_lbfgs.x, dtype=float).flatten()
                    if res_params.size == 2 * N:
                        # Evaluate the hard min area for this result
                        points = params_to_points(res_params)
                        min_area = calculate_min_area(points)
                        
                        if min_area > current_val:
                            current_val = min_area
                            current_params = res_params

            except Exception as e:
                pass
                
            if time.time() > deadline - 5:
                break
                
        best_params = current_params
        
    # Final safety check
    if best_params is None:
        # Generate valid random points
        u = np.random.rand(N)
        w = np.random.rand(N)
        v = w * (1 - u)
        x = u + 0.5 * v
        y = (3.0 ** 0.5) / 2.0 * v
        best_params = np.column_stack((u, w)).flatten() # Wait, params_to_points expects u,w interleaved or separate?
        # params_to_points expects params[0::2]=u, params[1::2]=w.
        # So we need to interleave u and w.
        params = np.empty(2*N)
        params[0::2] = u
        params[1::2] = w
        best_params = params

    return params_to_points(best_params)

# EVOLVE-BLOCK-END
