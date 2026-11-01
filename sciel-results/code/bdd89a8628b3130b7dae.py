import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_deg_fast(config, grid_size=20000):
    """
    Fast covering radius estimation using dense grid sampling.
    """
    grid = _fib_sample(grid_size)
    dots = grid @ config.T
    min_dots = np.min(dots, axis=1)
    max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.degrees(np.max(max_angle))

def _covering_radius_deg_hull(config):
    """
    Compute covering radius using convex hull facets (more accurate but slower).
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return _covering_radius_deg_fast(config, 30000)
    
    max_angle = 0.0
    for simplex in hull.simplices:
        pts = config[simplex]
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            continue
        
        c = c / c_norm
        
        # Check if circumcenter is valid
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        if np.max(angles) - np.min(angles) > 1e-6:
            # Worst point is a vertex
            max_in_facet = np.max(angles)
        else:
            max_in_facet = angles[0]
        
        # Check nearest config point to circumcenter
        dots_all = np.clip(config @ c, -1.0, 1.0)
        min_angle_to_c = np.arccos(np.max(dots_all))
        
        if min_angle_to_c > max_angle:
            max_angle = min_angle_to_c
    
    return np.degrees(max_angle)

def _soft_max_objective(config, beta=50.0):
    """
    Smooth approximation of the covering radius using soft-max.
    Returns (objective_value, gradient) where objective is in radians.
    """
    # Use a moderate grid for gradient computation
    grid_size = 5000
    grid = _fib_sample(grid_size)
    
    # Compute dot products: shape (grid_size, N)
    dots = grid @ config.T
    
    # For each grid point, find the max dot product (closest config point)
    # Use soft-max to smooth the max operation
    # log-sum-exp trick for numerical stability
    max_dots = np.max(dots, axis=1)
    dots_shifted = dots - max_dots[:, np.newaxis]
    exp_dots = np.exp(beta * dots_shifted)
    sum_exp = np.sum(exp_dots, axis=1)
    
    # Soft-max approximation of max
    soft_max_dots = max_dots + np.log(sum_exp) / beta
    
    # The objective is -soft_max_dots (we want to maximize the min dot product)
    # Or equivalently, arccos of the min dot product
    # Let's use: objective = arccos(soft_max_dots)
    # But arccos is not smooth at boundaries, so let's work with -soft_max_dots
    
    # Gradient: d/dconfig of -soft_max_dots
    # soft_max_dots[i] = max_dots[i] + log(sum_j exp(beta*(dots[i,j] - max_dots[i]))) / beta
    # d soft_max_dots[i] / d config[k] = (1/beta) * sum_j exp(beta*(dots[i,j] - max_dots[i])) * (grid[i] @ e_k) / sum_j exp(beta*(dots[i,j] - max_dots[i]))
    # where e_k is the k-th standard basis vector
    
    # More efficiently:
    # weights[i,j] = exp(beta*(dots[i,j] - max_dots[i])) / sum_exp[i]
    # grad_config[k] = -sum_i weights[i,:] @ grid[i]_k ... let me think again
    
    # Actually, let's compute the gradient properly:
    # f(config) = -arccos(soft_max_dots)
    # But let's just minimize -soft_max_dots directly (larger is better for covering)
    # So we maximize soft_max_dots, or minimize -soft_max_dots
    
    # Let's define objective = -soft_max_dots (to minimize)
    objective = -np.mean(soft_max_dots)
    
    # Gradient of soft_max_dots[i] w.r.t. config[k]:
    # dots[i,j] = grid[i] @ config[j]
    # d dots[i,j] / d config[k] = grid[i]_k if j==k, else 0
    # 
    # soft_max_dots[i] = max_dots[i] + (1/beta) * log(sum_j exp(beta*(dots[i,j] - max_dots[i])))
    # 
    # Let S[i] = sum_j exp(beta*(dots[i,j] - max_dots[i]))
    # d soft_max_dots[i] / d config[k] = (1/beta) * (1/S[i]) * sum_j exp(beta*(dots[i,j] - max_dots[i])) * d dots[i,j]/d config[k]
    # = (1/beta) * (1/S[i]) * exp(beta*(dots[i,k] - max_dots[i])) * grid[i]_k
    # 
    # So grad_objective[k] = -sum_i (1/beta) * exp(beta*(dots[i,k] - max_dots[i])) / S[i] * grid[i]_k
    
    # Compute weights: weights[i,j] = exp(beta*(dots[i,j] - max_dots[i])) / S[i]
    weights = exp_dots / sum_exp[:, np.newaxis]  # shape (grid_size, N)
    
    # grad[k] = -(1/beta) * sum_i weights[i,k] * grid[i]
    # = -(1/beta) * (weights.T @ grid)
    grad = -(1.0 / beta) * (weights.T @ grid)  # shape (N, 3)
    
    # Scale to be consistent with angle-based objective
    # The derivative of arccos(x) is -1/sqrt(1-x^2), so we need to account for that
    # But for simplicity, let's just use the soft-max gradient
    
    return objective, grad

def _lloyd_step(config, grid_size=8000):
    """
    Lloyd-type step: move each point to the centroid of its Voronoi cell.
    On the sphere, this means moving to the spherical centroid of the nearest grid points.
    """
    grid = _fib_sample(grid_size)
    dots = grid @ config.T  # (grid_size, N)
    nearest = np.argmax(dots, axis=1)  # (grid_size,)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = nearest == i
        if np.sum(mask) > 0:
            # Spherical centroid: normalize the sum of unit vectors
            centroid = np.sum(grid[mask], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_config[i] = centroid / norm
            else:
                new_config[i] = config[i]
        else:
            new_config[i] = config[i]
    
    return new_config

def _projected_gradient_polish(config, deadline, beta_start=30.0, beta_end=100.0, 
                                 lr_start=0.05, lr_end=0.001, max_iter=200):
    """
    Projected gradient descent with adaptive learning rate and increasing beta.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_fast(config, 10000)
    
    n_iter = 0
    while time.time() < deadline - 5.0 and n_iter < max_iter:
        # Interpolate beta and learning rate
        t = n_iter / max_iter
        beta = beta_start + (beta_end - beta_start) * t
        lr = lr_start + (lr_end - lr_start) * t
        
        # Get objective and gradient
        obj, grad = _soft_max_objective(config, beta)
        
        # Take gradient step
        config_new = config - lr * grad
        
        # Project back to sphere
        config_new = np.array([_normalize(p) for p in config_new])
        
        # Check improvement
        new_score = _covering_radius_deg_fast(config_new, 5000)
        if new_score < best_score:
            best_score = new_score
            best_config = config_new.copy()
            config = config_new
        else:
            # If no improvement, try smaller step
            config_new = config - lr * 0.5 * grad
            config_new = np.array([_normalize(p) for p in config_new])
            new_score = _covering_radius_deg_fast(config_new, 5000)
            if new_score < best_score:
                best_score = new_score
                best_config = config_new.copy()
                config = config_new
        
        n_iter += 1
    
    return best_config

def _local_optimize(config, deadline, verbose=False):
    """
    Local optimization combining Lloyd steps and projected gradient descent.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_fast(config, 10000)
    
    # Phase 1: Lloyd iterations for global structure
    lloyd_deadline = deadline - 30.0
    while time.time() < lloyd_deadline:
        config_new = _lloyd_step(config)
        new_score = _covering_radius_deg_fast(config_new, 8000)
        
        if new_score < best_score - 1e-6:
            best_score = new_score
            best_config = config_new.copy()
            config = config_new
        else:
            # Try a blend
            for alpha in [0.5, 0.3, 0.1]:
                config_blend = alpha * config_new + (1 - alpha) * config
                config_blend = np.array([_normalize(p) for p in config_blend])
                score_blend = _covering_radius_deg_fast(config_blend, 5000)
                if score_blend < best_score - 1e-6:
                    best_score = score_blend
                    best_config = config_blend.copy()
                    config = config_blend
                    break
            else:
                break
    
    # Phase 2: Projected gradient polish
    if time.time() < deadline - 10.0:
        config = _projected_gradient_polish(config, deadline)
        score = _covering_radius_deg_fast(config, 10000)
        if score < best_score:
            best_score = score
            best_config = config.copy()
    
    # Phase 3: Final Lloyd + gradient hybrid
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            config_lloyd = _lloyd_step(config)
            config_grad = _projected_gradient_polish(config, min(deadline - 5.0, time.time() + 10.0), 
                                                       beta_start=50.0, beta_end=150.0,
                                                       lr_start=0.02, lr_end=0.005, max_iter=50)
            
            # Try both
            for cand in [config_lloyd, config_grad]:
                score = _covering_radius_deg_fast(cand, 10000)
                if score < best_score - 1e-6:
                    best_score = score
                    best_config = cand.copy()
                    config = cand
    
    # Final refinement with hull-based method
    if time.time() < deadline - 1.0:
        final_score = _covering_radius_deg_hull(best_config)
        if final_score < best_score:
            best_score = final_score
    
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run local optimization
    config = _local_optimize(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
