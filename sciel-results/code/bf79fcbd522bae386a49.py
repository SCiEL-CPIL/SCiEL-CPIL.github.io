import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triples for efficiency
_TRIPLES = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _TRIPLES.append((i, j, k))
_TRIPLES = np.array(_TRIPLES)  # shape (165, 3)

def min_triangle_area(pts):
    """
    Vectorized calculation of the minimum area of triangles formed by N points.
    pts: (N, 2)
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    xi = x[_TRIPLES[:, 0]]
    yi = y[_TRIPLES[:, 0]]
    xj = x[_TRIPLES[:, 1]]
    yj = y[_TRIPLES[:, 1]]
    xk = x[_TRIPLES[:, 2]]
    yk = y[_TRIPLES[:, 2]]
    
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))

def softmin_triangle_area(pts, T=0.01):
    """
    Vectorized calculation of the soft-minimum area of triangles formed by N points.
    Uses log-sum-exp trick for stability.
    pts: (N, 2)
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    xi = x[_TRIPLES[:, 0]]
    yi = y[_TRIPLES[:, 0]]
    xj = x[_TRIPLES[:, 1]]
    yj = y[_TRIPLES[:, 1]]
    xk = x[_TRIPLES[:, 2]]
    yk = y[_TRIPLES[:, 2]]
    
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    # To avoid underflow/overflow, shift by max area
    a_max = np.max(areas)
    # exp(-(areas - a_max)/T) = exp(-areas/T) * exp(a_max/T)
    # log(sum(exp(-areas/T))) = a_max/T + log(sum(exp(-(areas-a_max)/T)))
    # softmin = -T * (a_max/T + log(sum(exp(-(areas-a_max)/T))))
    # softmin = -a_max - T * log(sum(exp(-(areas-a_max)/T)))
    
    exp_terms = np.exp(-(areas - a_max) / T)
    sum_exp = np.sum(exp_terms)
    soft_min_val = -a_max - T * np.log(sum_exp)
    
    return float(soft_min_val)

def _random_point_in_triangle(rng):
    """Generate a random point uniformly in the triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = v * S3H
    return np.array([x, y])

def project_to_triangle(pts):
    """
    Project points to be inside the triangle.
    """
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    
    for _ in range(10):
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, S3H)
        
        x_min = y / S3
        x_max = 1 - y / S3
        
        x_new = np.clip(x, x_min, x_max)
        
        if np.allclose(x, x_new, atol=1e-12):
            break
        x = x_new
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    T = 0.005  # Temperature for soft-min
    
    def objective(flat):
        pts = flat.reshape(-1, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalty for being outside the triangle
        # g1 = y >= 0
        # g2 = S3*x - y >= 0
        # g3 = S3*(1-x) - y >= 0
        g1 = y
        g2 = S3 * x - y
        g3 = S3 * (1 - x) - y
        
        viol = np.maximum(0, -g1) + np.maximum(0, -g2) + np.maximum(0, -g3)
        pen = np.sum(viol ** 2)
        
        # If significantly outside, return large penalty to guide back
        if np.any(viol > 0.1):
            return 100.0 + pen
            
        # Soft min area
        try:
            sm_area = softmin_triangle_area(pts, T)
        except:
            return 100.0 + pen
            
        # We want to maximize the min area, so minimize -softmin_area
        # But softmin is an approximation of min.
        # If all areas are large, softmin is large. We want to increase it.
        # So we minimize -softmin_area.
        return -sm_area + 10.0 * pen

    # Bounds for L-BFGS-B
    bounds_reordered = []
    for i in range(N):
        bounds_reordered.append((0, 1))      # x_i
        bounds_reordered.append((0, S3H))    # y_i
    
    # Generate diverse initial configurations
    def generate_initial_configs():
        configs = []
        
        # 1. Symmetric hexagonal-like pattern
        pts_sym = np.array([
            [0.5, S3H * 0.333],
            [0.5, S3H * 0.167],
            [0.5, S3H * 0.5],
            [0.333, S3H * 0.25],
            [0.667, S3H * 0.25],
            [0.333, S3H * 0.42],
            [0.667, S3H * 0.42],
            [0.25, S3H * 0.1],
            [0.75, S3H * 0.1],
            [0.5, 0.05],
            [0.5, S3H * 0.6]
        ])
        configs.append(pts_sym)
        
        # 2. Grid-like pattern
        grid_pts = []
        for row in range(4):
            for col in range(3):
                if len(grid_pts) < N:
                    t = (row + 0.5) / 4
                    s = (col + 0.5) / 3
                    x = 0.5 + (s - 0.5) * (1 - t)
                    y = t * S3H * (1 - 0.5 * t)
                    grid_pts.append([x, y])
        configs.append(np.array(grid_pts[:N]))
        
        # 3-10. Random configurations
        for _ in range(8):
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            configs.append(pts)
            
        return configs

    configs = generate_initial_configs()
    
    for i, init_pts in enumerate(configs):
        if time.time() > deadline - 5:
            break
        
        init_pts = project_to_triangle(init_pts.copy())
        
        # Evaluate initial
        area = min_triangle_area(init_pts)
        if area > best_area:
            best_area = area
            best_pts = init_pts.copy()
        
        # Optimize with L-BFGS-B
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_reordered,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            opt_pts = result.x.reshape(-1, 2)
            opt_pts = project_to_triangle(opt_pts)
            
            area = min_triangle_area(opt_pts)
            
            # Verify validity
            x = opt_pts[:, 0]
            y = opt_pts[:, 1]
            if (np.all(y >= -1e-6) and np.all(S3 * x - y >= -1e-6) and 
                np.all(S3 * (1 - x) - y >= -1e-6)):
                if area > best_area:
                    best_area = area
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # Additional random restarts if time permits
    while time.time() < deadline - 10:
        init_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        init_pts = project_to_triangle(init_pts.copy())
        
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_reordered,
                options={'maxiter': 150, 'ftol': 1e-7}
            )
            
            opt_pts = result.x.reshape(-1, 2)
            opt_pts = project_to_triangle(opt_pts)
            
            area = min_triangle_area(opt_pts)
            
            x = opt_pts[:, 0]
            y = opt_pts[:, 1]
            if (np.all(y >= -1e-6) and np.all(S3 * x - y >= -1e-6) and 
                np.all(S3 * (1 - x) - y >= -1e-6)):
                if area > best_area:
                    best_area = area
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = np.array([
            [0.5, S3H * 0.333],
            [0.5, S3H * 0.167],
            [0.5, S3H * 0.5],
            [0.333, S3H * 0.25],
            [0.667, S3H * 0.25],
            [0.333, S3H * 0.42],
            [0.667, S3H * 0.42],
            [0.25, S3H * 0.1],
            [0.75, S3H * 0.1],
            [0.5, 0.05],
            [0.5, S3H * 0.6]
        ])
        best_pts = project_to_triangle(best_pts)
        
    return best_pts
