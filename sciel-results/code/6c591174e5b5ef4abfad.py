import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area_vec(points, beta=50.0):
    """
    Differentiable soft-minimum of triangle areas using log-sum-exp.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area))))
    soft_min_val = -(log_sum + beta * max_area) / beta
    
    return soft_min_val

def params_to_points(params):
    """
    Maps 2*N parameters in [0,1] to N points in the equilateral triangle.
    Uses a smooth mapping from the unit square to the triangle:
    b = p1 * (1 - p2)
    c = p2 * (1 - p1)
    x = b + 0.5 * c
    y = H * c
    This ensures points are strictly inside or on the boundary of the triangle.
    """
    params = np.array(params).reshape(N, 2)
    p1 = params[:, 0]
    p2 = params[:, 1]
    
    b = p1 * (1.0 - p2)
    c = p2 * (1.0 - p1)
    
    x = b + 0.5 * c
    y = H * c
    
    return np.column_stack((x, y))

def objective_exact(params):
    """
    Exact objective for minimization (we minimize negative area).
    """
    points = params_to_points(params)
    return -min_triangle_area_vec(points)

def objective_soft(params, beta=50.0):
    """
    Soft objective for gradient-based refinement.
    """
    points = params_to_points(params)
    return -soft_min_area_vec(points, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Global Search using Differential Evolution
    # Bounds: each of the 2*N parameters is in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -np.inf
    
    # Run Differential Evolution
    # popsize=10, maxiter=50 is a reasonable trade-off for 22 dimensions
    # tol=0.01 to stop early if converged
    try:
        de_result = differential_evolution(
            objective_exact, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=0.01,
            polish=False # We will do our own polish
        )
        
        if time.time() < deadline - 10.0:
            current_params = de_result.x
            current_score = -min_triangle_area_vec(params_to_points(current_params))
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                
    except Exception as e:
        print(f"DE failed: {e}")
    
    # 2. Local Refinement using L-BFGS-B on Soft Objective
    # Start from the best DE result or a grid point if DE failed
    if best_params is None:
        # Fallback: Grid-like initialization
        best_params = np.linspace(0.1, 0.9, N * 2).reshape(N, 2).flatten()
        best_score = -min_triangle_area_vec(params_to_points(best_params))
        
    current_params = best_params.copy()
    
    # Try a few beta values for refinement
    for beta in [10.0, 50.0, 100.0]:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                lambda p: objective_soft(p, beta),
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < 1e10:
                new_params = res.x
                new_score = -min_triangle_area_vec(params_to_points(new_params))
                
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params.copy()
                    current_params = new_params.copy()
                    
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
            break

    # Final check
    if best_params is not None:
        points = params_to_points(best_params)
        # Ensure no NaNs or Infs
        if np.all(np.isfinite(points)):
            return points
    
    # Fallback return: Centroid (should not happen, but safe)
    centroid = np.full((N, 2), [0.5, H/3])
    return centroid
