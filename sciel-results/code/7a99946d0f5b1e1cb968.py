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

def _normalize_batch(pts):
    """Normalize batch of vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_circumcenters_and_radii(config):
    """
    Compute circumcenters of convex hull facets and their angular distance
    to the nearest configuration point.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    c = np.cross(v1, v2)
    c_norms = np.linalg.norm(c, axis=1)
    
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([], dtype=int)
        
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    
    # Compute angular distance from each circumcenter to the nearest config point
    dots = c @ config.T  # (F, N)
    dots = np.clip(dots, -1.0, 1.0)
    nearest_idx = np.argmax(dots, axis=1)
    nearest_dots = dots[np.arange(len(c)), nearest_idx]
    radii = np.arccos(nearest_dots)
    
    return c, radii, nearest_idx

def _lloyd_step(cfg, grid):
    """One Lloyd's algorithm step (vectorized)."""
    dots = grid @ cfg.T  # (M, N)
    max_idx = np.argmax(dots, axis=1)  # (M,)
    
    new_cfg = np.zeros_like(cfg)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_cfg[i] = _normalize(center)
        else:
            new_cfg[i] = _normalize(np.random.randn(3))
    return new_cfg

def _subgradient_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using subgradient method on the covering radius.
    """
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.02
    stagnation = 0
    max_stagnation = 100
    weight_sharpness = 50.0
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-10:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        # Compute forces: move points towards the circumcenters that define the max radius
        forces = np.zeros_like(config)
        # Exponential weighting to focus on the worst circumcenters
        weights = np.exp(-weight_sharpness * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            # Vector from point i to circumcenter k
            v = c[k] - config[i]
            # Project onto tangent space at config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        # Apply step
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        # Evaluate new configuration
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-9:
            config = new_config
            current_R = new_R
            # Increase step size slightly if improving significantly
            if current_R < best_R:
                step_size = min(step_size * 1.02, 0.05)
        else:
            # Reject move, reduce step size
            step_size *= 0.95
            if step_size < 0.001:
                break
                
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish with very small steps and sharp weights.
    """
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.002
    stagnation = 0
    max_stagnation = 200
    weight_sharpness = 200.0
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-12:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        forces = np.zeros_like(config)
        weights = np.exp(-weight_sharpness * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            v = c[k] - config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-12:
            config = new_config
            current_R = new_R
        else:
            step_size *= 0.9
            if step_size < 0.0001:
                break
                
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin

    # Initial configuration: Fibonacci spiral
    config = _fib_sample(N)
    
    # Phase 1: Lloyd's algorithm to establish good Voronoi structure
    M = 10000
    grid = _fib_sample(M)
    
    # Run Lloyd's for a fixed number of iterations or until time limit
    # 100 iterations should be fast enough
    for iteration in range(100):
        if time.time() > deadline - 60:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
        
    # Phase 2: Subgradient optimization
    if time.time() < deadline - 20:
        config = _subgradient_optimize(config, deadline, min_time_remaining=15.0)
        
    # Phase 3: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
        
    # Final normalization and validation
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
