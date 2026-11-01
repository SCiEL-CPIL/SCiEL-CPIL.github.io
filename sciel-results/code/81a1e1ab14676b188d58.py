import numpy as np
import time

N = 50

def _get_radius(points):
    """
    Calculate the maximum radius for the given points.
    points: (N, 2) array
    Returns: radius (float)
    """
    if not np.all(np.isfinite(points)):
        return -np.inf
        
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    x_dists = 0.5 - np.abs(points[:, 0])
    y_dists = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([x_dists, y_dists]))
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _hexagonal_init(N):
    """
    Generate an initial hexagonal packing configuration.
    """
    # Create a hexagonal lattice
    # We want to fit N points.
    # Approximate number of rows/cols
    # For N=50, roughly 7x7 or 5x10.
    # Let's try to generate a larger hexagonal grid and pick N points closest to center.
    
    # Hexagonal lattice basis vectors
    a = 1.0 # spacing
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    
    # Generate points in a grid of indices
    max_idx = 15
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:N]]
    
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Scale to fit within [-0.4, 0.4] initially (leaving margin)
    max_extent = np.max(np.abs(points))
    if max_extent > 0:
        points *= (0.4 / max_extent)
        
    return points

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Initial guess
    points = _hexagonal_init(N)
    
    # Initial temperature
    # Estimate typical distance
    # Let's start with T such that moves are ~0.1
    T = 0.1
    T_min = 1e-8
    cooling_factor = 0.9995
    
    best_points = points.copy()
    best_radius = _get_radius(points)
    
    current_radius = best_radius
    current_points = points.copy()
    
    # Simulated Annealing
    step = 0
    while time.time() < deadline:
        # Pick a random index
        idx = np.random.randint(N)
        
        # Generate a random move
        # Step size proportional to T
        delta = np.random.normal(0, T, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Check if it's valid (within bounds, though SA allows violations with penalty)
        # We use a penalty method or just rely on the radius calculation returning small values for invalid configs.
        # _get_radius returns min(dist/2, wall_dist). If outside, wall_dist is negative.
        # So radius becomes negative. This is a good penalty.
        
        new_radius = _get_radius(new_points)
        
        delta_E = -new_radius - (-current_radius) # Change in energy (Energy = -Radius)
        
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        
        # Cool down
        T *= cooling_factor
        if T < T_min:
            T = T_min
            
        step += 1
        
        # Periodically restart or perturb if stuck?
        # For now, simple SA.
        
    # Final polish with Nelder-Mead on the best points found
    # Since Nelder-Mead is fast for small improvements, we can try it.
    # But it might be slow. Let's just return the SA result if time is tight.
    
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def _neg_radius(points_flat):
                pts = points_flat.reshape(N, 2)
                return -_get_radius(pts)
            
            res = minimize(_neg_radius, best_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < -best_radius:
                best_radius = -res.fun
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass

    # Ensure validity
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
