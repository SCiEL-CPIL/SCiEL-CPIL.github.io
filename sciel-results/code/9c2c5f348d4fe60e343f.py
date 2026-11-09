import numpy as np
from scipy.optimize import differential_evolution, minimize
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
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area_vec(points, beta=100.0):
    """
    Differentiable soft-minimum of triangle areas using log-sum-exp.
    Returns a smooth approximation of -min(areas).
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # To avoid overflow, subtract max
    max_area = np.max(areas)
    # Ensure numerical stability
    if max_area < 1e-12:
        return -max_area
    
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area))))
    soft_min_val = -(log_sum + beta * max_area) / beta
    
    return soft_min_val

def params_to_points_symmetric(params):
    """
    Maps 11 parameters to 11 points in the triangle, enforcing reflection symmetry.
    params[0]: y-coordinate of the central point (x=0.5)
    params[1:]: x, u for 5 pairs. x in [0, 0.5], u in [0, 1]. y = u * sqrt(3) * x.
    
    Note: The symmetry is about x=0.5.
    Point 0: (0.5, y0)
    Points 1,2: (x1, y1), (1-x1, y1) where y1 = u1 * sqrt(3) * x1
    ...
    Points 9,10: (x5, y5), (1-x5, y5) where y5 = u5 * sqrt(3) * x5
    """
    params = np.array(params)
    points = np.zeros((N, 2))
    
    # Central point
    points[0, 0] = 0.5
    points[0, 1] = params[0]
    
    # 5 pairs
    for i in range(5):
        x = params[1 + 2*i]
        u = params[2 + 2*i]
        y = u * np.sqrt(3) * x
        
        # Pair 1: (x, y)
        points[1 + 2*i, 0] = x
        points[1 + 2*i, 1] = y
        
        # Pair 2: (1-x, y)
        points[2 + 2*i, 0] = 1.0 - x
        points[2 + 2*i, 1] = y
        
    return points

def objective_symmetric(params):
    """
    Objective function to minimize (negative of min triangle area) for symmetric configuration.
    """
    points = params_to_points_symmetric(params)
    return -min_triangle_area_vec(points)

def objective_soft_symmetric(params, beta=100.0):
    """
    Differentiable objective using soft-min for gradient-based optimization.
    """
    points = params_to_points_symmetric(params)
    return soft_min_area_vec(points, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Initial random start
    rng = np.random.default_rng(42)
    
    # Bounds for DE
    # y_0 in [0, H]
    # x_i in [0, 0.5]
    # u_i in [0, 1]
    bounds = [(0.0, H)]
    for _ in range(5):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, 1.0))
        
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    if time.time() < deadline - 60.0:
        try:
            result = differential_evolution(
                objective_symmetric, 
                bounds, 
                maxiter=150, 
                popsize=20, 
                seed=42, 
                tol=1e-6,
                polish=False
            )
            best_params = result.x
            best_score = -result.fun
            
        except Exception as e:
            print(f"DE failed: {e}")
            # Fallback to random
            best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            best_score = -objective_symmetric(best_params)

    # Phase 2: Local Refinement using Gradient-based method on Soft-Min
    # We use the soft-min objective to allow gradients to guide the optimization
    # We will try multiple beta values to balance smoothness and accuracy
    if best_params is not None and time.time() < deadline - 10.0:
        current_params = best_params.copy()
        current_score = best_score
        
        # Try refining with different beta values
        betas = [50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            # Define the specific soft objective for this beta
            def obj_soft_p(params):
                return objective_soft_symmetric(params, beta)
            
            try:
                # Use L-BFGS-B as it handles bounds well and is efficient for medium dimensions
                result = minimize(
                    obj_soft_p,
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Evaluate the HARD objective to see if it actually improved
                hard_score = -objective_symmetric(result.x)
                
                if hard_score > current_score:
                    current_score = hard_score
                    current_params = result.x.copy()
                    
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    # Fallback if nothing worked
    if best_params is None:
        best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
    
    # Convert best parameters to points
    points = params_to_points_symmetric(best_params)
    
    # Final sanity check: ensure points are within bounds (they should be by construction)
    # x in [0, 1], y in [0, H]
    # The construction ensures x in [0, 0.5] for left points and [0.5, 1] for right.
    # y = u * sqrt(3) * x. Since x <= 0.5 and u <= 1, y <= 0.5 * sqrt(3) = H.
    # Also y >= 0.
    
    return points
