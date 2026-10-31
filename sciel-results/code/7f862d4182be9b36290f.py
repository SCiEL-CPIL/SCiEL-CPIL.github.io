"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Vectorized for speed.
    points: (N, 2) array
    """
    # Generate all combinations of indices (i, j, k) with i < j < k
    # N=11 -> 165 combinations
    if N < 3:
        return 0.0
        
    # Precompute indices for all unique triples
    # This is constant for N=11, so we can cache it or compute once.
    # For clarity, compute inside, it's very fast for 165 items.
    idx = np.indices((N, N, N))
    # Mask where i < j < k
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i_idx = idx[0][mask]
    j_idx = idx[1][mask]
    k_idx = idx[2][mask]
    
    # Get coordinates
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Vectors
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product z-component (2D cross)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-9
    
    # Constraints:
    # y >= 0
    # y <= h
    # y <= sqrt(3) * x  => x >= y/sqrt(3)
    # y <= sqrt(3) * (1 - x) => x <= 1 - y/sqrt(3)
    
    y = np.clip(y, eps, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min_left + eps, x_max_right - eps)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _simulated_annealing(start_points, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    points = np.array(start_points, dtype=float)
    points = _clamp_all(points)
    current_score = _min_triangle_area_vec(points)
    
    best_points = points.copy()
    best_score = current_score
    
    # Parameters
    T = 0.01 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    step_size = 0.05 # Initial step size for perturbation
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        
        # Perturb with Gaussian noise
        noise = np.random.normal(0, step_size, 2)
        new_point = points[idx] + noise
        
        # Clamp the new point
        new_point = _clamp_to_triangle(new_point)
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        new_score = _min_triangle_area_vec(points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Revert
            points[idx] = old_point
            
        # Cool down
        T *= alpha
        # Slightly reduce step size to allow fine tuning later
        step_size *= 0.9999
        
    return best_points, best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    h = np.sqrt(3) / 2
    configs = []
    
    # 1. Symmetric: Vertices, Centroid, Midpoints of sides, etc.
    # A good packing for 11 points often involves the vertices and a central cluster.
    # Let's try: 3 vertices, 3 midpoints, 3 near-center, 2 more.
    sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3], # Centroid
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4]
    ])
    configs.append(sym)
    
    # 2. Random configurations
    rng = np.random.default_rng(123)
    for i in range(10):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        pts = _clamp_all(pts)
        configs.append(pts)
        
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 130 # 20s safety margin
    
    best_points = None
    best_score = -1.0
    
    configs = _generate_initial_configs()
    
    # We will run SA for each config, allocating time based on remaining budget.
    # We can do a few fast runs, then one long run on the best.
    
    # First pass: Quick SA runs to find a good basin
    n_configs = len(configs)
    time_per_config = 5.0 # 5 seconds per config initially
    
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        # If time is short, reduce time per config
        current_time_limit = min(time_per_config, time_left / (n_configs - i))
        
        if current_time_limit < 1.0:
            break
            
        try:
            opt_pts, score = _simulated_annealing(config, time_limit=current_time_limit)
            if score > best_score:
                best_score = score
                best_points = opt_pts
        except Exception:
            continue
            
    # Second pass: Long SA run on the best found so far
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 0:
            try:
                opt_pts, score = _simulated_annealing(best_points, time_limit=remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = configs[0]
        
    return best_points

# EVOLVE-BLOCK-END
