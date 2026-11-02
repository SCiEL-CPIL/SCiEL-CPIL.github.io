import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0  # Area of the unit-side equilateral triangle
DEADLINE = 110.0      # Safety margin from the 120s limit

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    Vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    u is weight for V1, v is weight for V2, w=1-u-v is weight for V0.
    x = u*1 + v*0.5
    y = v*sqrt(3)/2
    """
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by three points.
    points: (N, 2) array
    """
    n = len(points)
    min_a = float('inf')
    # Pre-compute coordinate arrays for speed
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * | (xj-xi)(yk-yi) - (xk-xi)(yj-yi) |
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                if area < min_a:
                    min_a = area
    return min_a

def soft_min_objective(params, beta):
    """
    Objective function for local optimization (L-BFGS-B).
    params: flat array of length 2*N, containing (u_0, v_0, u_1, v_1, ...)
    Returns: -soft_min_area
    """
    u = params[0::2]
    v = params[1::2]
    
    # Ensure points are strictly inside to avoid numerical issues at boundaries
    # If u+v > 1, scale down (should not happen with bounds, but safe)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u = u.copy()
        v = v.copy()
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
        
    points = bary_to_cart(u, v)
    
    # Compute all triangle areas
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft min calculation
    # We want to maximize min(areas).
    # Soft min: S = -1/beta * log(sum(exp(-beta * areas)))
    # To make it stable:
    m = np.min(areas)
    if m < 1e-12:
        m = 1e-12
    # log_sum_exp(-beta * (areas - m))
    x_val = -beta * (areas - m)
    log_sum_exp = np.log(np.sum(np.exp(x_val)))
    soft_min_val = m - (1.0 / beta) * log_sum_exp
    
    # We minimize the negative soft min
    return -soft_min_val

def raw_objective(params):
    """
    Objective for Differential Evolution.
    Returns: -min_area (since DE minimizes)
    """
    u = params[0::2]
    v = params[1::2]
    
    # Handle invalid barycentric coords (u+v > 1) by scaling
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u = u.copy()
        v = v.copy()
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
        
    points = bary_to_cart(u, v)
    min_a = compute_min_area(points)
    
    # Add a small penalty if points are too close to boundary to encourage interior
    # This helps avoid degenerate solutions where points stick to edges
    boundary_dist = np.min(1.0 - (u + v), axis=0)
    penalty = 0.01 * np.sum(boundary_dist**2)
    
    return -(min_a - penalty)

def construct():
    start_time = time.time()
    
    # Bounds for barycentric coords: u in [0, 1], v in [0, 1]
    # Note: The constraint u+v <= 1 is handled by scaling in the objective,
    # but DE works better with explicit bounds. 
    # However, the feasible region is a triangle. 
    # Standard DE bounds are hyper-rectangles. 
    # We can define bounds as [0, 1] for both and rely on the scaling/penalty.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -np.inf  # We want to maximize min_area, so we track -min_area for minimizers, or min_area for comparison
    
    # --- Phase 1: Global Search with Differential Evolution ---
    # DE is robust but slow. We limit iterations and population.
    # 22 variables. popsize=20, maxiter=50 -> 1000 evaluations.
    # Each eval is ~165 area calcs. Very fast.
    
    try:
        if time.time() - start_time < DEADLINE - 30:
            de_res = differential_evolution(
                raw_objective, 
                bounds, 
                seed=42, 
                popsize=15, 
                maxiter=50, 
                tol=1e-6, 
                polish=False, # We will polish separately
                init='sobol'
            )
            
            if de_res.x is not None:
                best_params = de_res.x.copy()
                # Evaluate true score
                u = best_params[0::2]
                v = best_params[1::2]
                sum_uv = u + v
                mask = sum_uv > 1.0
                if np.any(mask):
                    u[mask] /= sum_uv[mask]
                    v[mask] /= sum_uv[mask]
                pts = bary_to_cart(u, v)
                current_min = compute_min_area(pts)
                best_score = current_min
                
    except Exception as e:
        # If DE fails, fall back to random
        pass
    
    # --- Phase 2: Local Refinement with Soft-Min L-BFGS-B ---
    # We try to refine the DE result and a few random restarts.
    
    def try_refine(x0, beta_schedule):
        nonlocal best_params, best_score
        current_x = x0.copy()
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                res = minimize(
                    soft_min_objective, 
                    current_x, 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res.success and res.x is not None:
                    current_x = res.x
            except:
                pass
        
        # Evaluate final result
        u = current_x[0::2]
        v = current_x[1::2]
        sum_uv = u + v
        mask = sum_uv > 1.0
        if np.any(mask):
            u[mask] /= sum_uv[mask]
            v[mask] /= sum_uv[mask]
        pts = bary_to_cart(u, v)
        min_area = compute_min_area(pts)
        
        if min_area > best_score:
            best_score = min_area
            best_params = current_x.copy()
            return pts # Return best points immediately if improved? No, keep optimizing
    
    # Beta schedule: Start low (smooth), end high (sharp min)
    betas = [5, 10, 20, 50, 100, 200]
    
    # Refine the DE result
    if best_params is not None:
        try_refine(best_params, betas)
    
    # Try a few random restarts if time permits
    np.random.seed(123)
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Random init in barycentric space
        u = np.random.rand(N)
        v = np.random.rand(N)
        sum_uv = u + v
        mask = sum_uv > 1.0
        if np.any(mask):
            u[mask] /= sum_uv[mask]
            v[mask] /= sum_uv[mask]
            
        x0 = np.zeros(2 * N)
        x0[0::2] = u
        x0[1::2] = v
        
        # Use a shorter beta schedule for restarts to save time
        try_refine(x0, [10, 50, 100])
        
    # Final check and return
    if best_params is None:
        # Absolute fallback: Grid or simple random
        # This should rarely happen
        u = np.random.rand(N)
        v = np.random.rand(N)
        sum_uv = u + v
        mask = sum_uv > 1.0
        if np.any(mask):
            u[mask] /= sum_uv[mask]
            v[mask] /= sum_uv[mask]
        best_params = np.zeros(2*N)
        best_params[0::2] = u
        best_params[1::2] = v
        
    # Convert best_params to Cartesian points
    u = best_params[0::2]
    v = best_params[1::2]
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
    
    points = bary_to_cart(u, v)
    
    # Final sanity check: ensure points are within bounds
    # x in [0, 1], y in [0, sqrt(3)/2]
    # Since u,v >= 0 and u+v <= 1, this is guaranteed.
    
    return points

