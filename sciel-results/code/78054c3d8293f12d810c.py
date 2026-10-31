import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
NUM_TRI = N * (N - 1) * (N - 2) // 6

def triangle_areas_vec(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2)
    returns: (NUM_TRI,)
    """
    n = points.shape[0]
    # Indices for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask].flatten()
    j_flat = j_idx[mask].flatten()
    k_flat = k_idx[mask].flatten()
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    # det = v1_x * v2_y - v1_y * v2_x
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(dets)
    return areas

def boundary_penalty(points):
    """
    Penalty for points outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    # d1: y >= 0
    d1 = y
    # d2: sqrt(3)*x - y >= 0
    d2 = SQRT3 * x - y
    # d3: sqrt(3) - sqrt(3)*x - y >= 0
    d3 = SQRT3 - SQRT3 * x - y
    
    pen = np.sum(np.maximum(0, -d1)**2) + \
          np.sum(np.maximum(0, -d2)**2) + \
          np.sum(np.maximum(0, -d3)**2)
    return pen

def softplus(x, beta=100.0):
    """
    Smooth approximation of max(0, x).
    """
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return (1.0 / beta) * np.log(1.0 + np.exp(beta * x))

def objective(flat_params):
    """
    Minimize -alpha + Penalty.
    flat_params: [x1, y1, ..., xN, yN, alpha]
    """
    P = flat_params[:2*N].reshape(N, 2)
    alpha = flat_params[2*N]
    
    areas = triangle_areas_vec(P)
    
    # Penalty for areas < alpha
    # We want max(0, alpha - area)^2
    violations = alpha - areas
    # Use softplus to make it differentiable
    # softplus(violations, beta) approximates max(0, violations)
    pen_areas = np.sum(softplus(violations, beta=50.0)**2)
    
    # Boundary penalty
    pen_bound = boundary_penalty(P)
    
    # Objective: Maximize alpha, so minimize -alpha
    # Weights: Alpha is the primary goal. Penalties should be large if violated.
    # If alpha is small, violations are small.
    # Scale: Typical area ~ 0.01. Alpha ~ 0.01.
    # Penalty coeff for areas: 1000
    # Penalty coeff for boundary: 1000
    
    return -alpha + 1000.0 * pen_areas + 1000.0 * pen_bound

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = -np.inf
    
    # Initializations
    # 1. Random uniform in triangle
    # 2. Grid-like or known good structures
    
    inits = []
    
    # Random restarts
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
        rng = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            pts[i, 0] = v * 1.0 + w * 0.5
            pts[i, 1] = w * SQRT3 / 2.0
        
        # Initial alpha: small value
        a0 = 0.001
        flat_init = np.concatenate([pts.flatten(), [a0]])
        inits.append(flat_init)
        
    # A structured init: Center + Ring
    # Center
    cx, cy = 0.5, SQRT3/6
    # Ring at radius r
    # We can try a few radii
    for r in [0.2, 0.3, 0.4]:
        pts = np.zeros((N, 2))
        pts[0] = [cx, cy]
        angles = np.linspace(0, 2*np.pi, N-1, endpoint=False)
        for i in range(1, N):
            pts[i, 0] = cx + r * np.cos(angles[i-1])
            pts[i, 1] = cy + r * np.sin(angles[i-1])
        # Check if inside, if not, scale down
        # For simplicity, just add it, optimizer will fix if penalty is high
        a0 = 0.001
        flat_init = np.concatenate([pts.flatten(), [a0]])
        inits.append(flat_init)

    # Optimization
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        try:
            # Use L-BFGS-B for gradient-based optimization
            # Bounds:
            # x in [0, 1], y in [0, SQRT3/2] is a loose box bound.
            # The triangle constraint is handled by penalty.
            # But box bounds help convergence.
            bounds = []
            for i in range(N):
                bounds.append((0, 1)) # x
                bounds.append((0, SQRT3/2)) # y
            bounds.append((0, 0.4)) # alpha upper bound (max possible is < 0.433/11 ~ 0.04)
            
            result = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e6:
                opt_flat = result.x
                opt_P = opt_flat[:2*N].reshape(N, 2)
                opt_alpha = opt_flat[2*N]
                
                # Verify
                areas = triangle_areas_vec(opt_P)
                min_area = np.min(areas)
                pen_b = boundary_penalty(opt_P)
                
                if pen_b < 1e-6:
                    # Score is min_area
                    # Note: The optimizer maximized alpha subject to alpha <= areas.
                    # So alpha should be close to min_area.
                    score = min_area
                    if score > best_score:
                        best_score = score
                        best_points = opt_P.copy()
                else:
                    # If boundary penalty is high, it's invalid
                    pass
                    
        except Exception as e:
            pass

    if best_points is None:
        # Fallback
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
