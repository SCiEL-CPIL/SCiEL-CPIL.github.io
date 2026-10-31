import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Triangle vertices in barycentric (u,v) space
# A=(0,0), B=(1,0), C=(0,1) in (u,v) -> Physical A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# Area scaling factor from barycentric (u,v) to physical is sqrt(3)/2

def get_min_bary_area(points_bary):
    """
    points_bary: (N, 2) array of (u, v) coordinates.
    Returns the minimum area of any triangle formed by these points in the (u,v) plane.
    """
    n = points_bary.shape[0]
    if n < 3:
        return 0.0
    
    min_area = float('inf')
    # Vectorized calculation is tricky for all triplets without huge memory usage if N is large, 
    # but N=11 is small. 11*10*9/6 = 165 triangles.
    
    # Extract coordinates
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    
    # Loop over triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det([Pj-Pi, Pk-Pi])|
                # det = (u_j - u_i)*(v_k - v_i) - (v_j - v_i)*(u_k - u_i)
                det = (u[j] - u[i]) * (v[k] - v_i) - (v[j] - v[i]) * (u[k] - u_i)
                area = 0.5 * abs(det)
                if area < min_area:
                    min_area = area
    
    return min_area

def objective_func(x):
    """
    x: flattened array of 2*N coordinates (u1, v1, u2, v2, ...)
    Returns a scalar to minimize. We want to maximize min_area, so we minimize -min_area.
    Also penalize points outside the triangle u>=0, v>=0, u+v<=1.
    """
    points = x.reshape(-1, 2)
    
    # Check validity
    u = points[:, 0]
    v = points[:, 1]
    
    # Violations
    viol_u = np.sum(np.maximum(0, -u) ** 2)
    viol_v = np.sum(np.maximum(0, -v) ** 2)
    viol_sum = np.sum(np.maximum(0, u + v - 1) ** 2)
    
    penalty = 100.0 * (viol_u + viol_v + viol_sum)
    
    if penalty > 1e-6:
        # If significantly outside, return a large value
        return 1e6 + penalty
    
    min_area = get_min_bary_area(points)
    
    # We want to maximize min_area, so minimize -min_area
    return -min_area

def random_points_in_triangle(rng):
    """
    Generate N random points in the barycentric triangle u>=0, v>=0, u+v<=1.
    """
    points = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        points.append([u, v])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_score = -float('inf')
    best_points = None
    
    # Multi-start optimization
    max_restarts = 500
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initial guess
        x0 = random_points_in_triangle(rng)
        
        # Try to improve with Nelder-Mead
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            x_opt = res.x
            score = -res.fun
            
            # Validate and calculate true score
            points_opt = x_opt.reshape(-1, 2)
            u = points_opt[:, 0]
            v = points_opt[:, 1]
            
            # Check if valid
            if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(u + v > 1 + 1e-6):
                continue
            
            # Clip to valid region to be safe
            points_opt[:, 0] = np.clip(points_opt[:, 0], 0, 1)
            points_opt[:, 1] = np.clip(points_opt[:, 1], 0, 1)
            # Fix u+v <= 1
            s = points_opt[:, 0] + points_opt[:, 1]
            mask = s > 1
            if np.any(mask):
                # Scale down
                scale = 1.0 / s[mask]
                points_opt[mask, 0] *= scale
                points_opt[mask, 1] *= scale
            
            min_area = get_min_bary_area(points_opt)
            
            if min_area > best_score:
                best_score = min_area
                best_points = points_opt.copy()
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to random if optimization failed completely
        best_points = random_points_in_triangle(rng).reshape(-1, 2)
        # Clip
        best_points[:, 0] = np.clip(best_points[:, 0], 0, 1)
        best_points[:, 1] = np.clip(best_points[:, 1], 0, 1)
        s = best_points[:, 0] + best_points[:, 1]
        mask = s > 1
        if np.any(mask):
            scale = 1.0 / s[mask]
            best_points[mask, 0] *= scale
            best_points[mask, 1] *= scale

    # Convert barycentric (u, v) to Cartesian (x, y)
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*B + v*C + (1-u-v)*A = u*(1,0) + v*(0.5, sqrt(3)/2)
    x = best_points[:, 0] + 0.5 * best_points[:, 1]
    y = (3 ** 0.5) / 2 * best_points[:, 1]
    
    return np.column_stack((x, y))
