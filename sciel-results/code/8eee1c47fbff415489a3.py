import numpy as np
from scipy.optimize import differential_evolution
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

def params_to_points_symmetric(params):
    """
    Maps 11 parameters to 11 points in the triangle, enforcing reflection symmetry.
    params[0]: y-coordinate of the central point (x=0.5)
    params[1:]: x, u for 5 pairs. x in [0, 0.5], u in [0, 1]. y = u * sqrt(3) * x.
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
    try:
        result = differential_evolution(
            objective_symmetric, 
            bounds, 
            maxiter=200, 
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

    # Phase 2: Local Refinement (Polish)
    if time.time() < deadline - 10.0:
        current_params = best_params.copy()
        current_score = best_score
        
        local_iters = 0
        max_local_iters = 5000
        step_size = 0.05
        
        while time.time() < deadline - 5.0 and local_iters < max_local_iters:
            local_iters += 1
            
            noise = rng.normal(0, step_size, size=len(bounds))
            new_params = current_params + noise
            
            # Clip to bounds
            for i, b in enumerate(bounds):
                new_params[i] = np.clip(new_params[i], b[0], b[1])
            
            new_score = -objective_symmetric(new_params)
            
            if new_score > current_score:
                current_score = new_score
                current_params = new_params
                step_size = max(0.001, step_size * 0.99)
                
        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    # Convert best parameters to points
    points = params_to_points_symmetric(best_params)
    
    return points
