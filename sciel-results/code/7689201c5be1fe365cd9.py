import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    """
    Compute the radius of the largest equal circles that can be placed 
    at the given centers in the square [-0.5, 0.5]^2.
    """
    # centers shape (N, 2)
    # 1. Distances between all pairs
    # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist = np.linalg.norm(diff, axis=2)
    # min_dist = np.min(dist[np.triu_indices(N, k=1)])
    
    # More efficient:
    # Compute pairwise squared distances
    # sq_dist = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
    # min_sq_dist = np.min(sq_dist[np.triu_indices(N, k=1)])
    # min_dist = np.sqrt(min_sq_dist)
    
    # Even faster for large N, but N=50 is small. Let's stick to simple vectorized norm.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask the diagonal and lower triangle
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    # 2. Distance to boundaries
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Dist to left: centers[:, 0] + 0.5
    # Dist to right: 0.5 - centers[:, 0]
    # Dist to bottom: centers[:, 1] + 0.5
    # Dist to top: 0.5 - centers[:, 1]
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    # Radius is half the minimum of pair distance and boundary distance
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    return r

def objective(centers_flat):
    """
    Objective function for minimization.
    Returns negative radius to maximize radius.
    """
    centers = centers_flat.reshape(N, 2)
    r = compute_radius(centers)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    best_radius = compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # Optimization parameters
    # Nelder-Mead is good for non-smooth functions
    options = {
        'maxiter': 1000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'disp': False
    }
    
    # We will try a few restarts with random perturbations
    # The grid is a strong local minimum, so we need to kick it out.
    # However, for 50 circles, the optimal packing is likely close to a distorted grid.
    # We will try 10-20 restarts.
    
    rng = np.random.default_rng(42)
    
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the initial grid
        # Small perturbations to break symmetry
        noise_scale = 0.01
        perturbed_centers = initial_centers + rng.normal(0, noise_scale, size=initial_centers.shape)
        
        # Ensure they stay within bounds roughly? 
        # Nelder-Mead can handle constraints poorly, so we might need to project or penalize.
        # The objective function naturally penalizes going outside because boundary distance becomes negative, 
        # but wait: if a point is outside, e.g., x > 0.5, d_right < 0. 
        # The radius would be negative. We want to maximize radius. 
        # So minimizing -radius means maximizing radius. 
        # If a point is outside, radius is negative, -radius is positive. 
        # We want to minimize -radius. So we want -radius to be as small (negative) as possible.
        # This means we want radius to be as large (positive) as possible.
        # So the objective function naturally pushes points inside.
        
        try:
            res = minimize(objective, perturbed_centers.flatten(), method='Nelder-Mead', options=options)
            new_centers = res.x.reshape(N, 2)
            new_radius = compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers
        except Exception:
            continue
            
    return best_centers
