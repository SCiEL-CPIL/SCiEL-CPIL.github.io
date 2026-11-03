import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def compute_radius(points):
    """
    Compute the radius of the largest equal circles that can be placed at 'points'.
    points: (N, 2) array
    Returns: max radius r
    """
    # Pairwise distances
    # dist[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Minimum pair distance
    # We need min over i < j
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            if dists[i, j] < min_pair_dist:
                min_pair_dist = dists[i, j]
    
    # Distance to walls
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Distance from point (x, y) to boundary is min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # Which is 0.5 - max(|x|, |y|) if we consider the square boundaries individually?
    # No, distance to the set of boundaries.
    # For a point inside, dist to left wall = x - (-0.5) = x + 0.5
    # dist to right wall = 0.5 - x
    # dist to bottom = y + 0.5
    # dist to top = 0.5 - y
    # The limiting distance is the minimum of these four.
    # min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    # = 0.5 - max(-x, x, -y, y) = 0.5 - max(|x|, |y|)
    
    abs_x = np.abs(points[:, 0])
    abs_y = np.abs(points[:, 1])
    max_coord = np.maximum(abs_x, abs_y)
    dist_to_wall = 0.5 - max_coord
    min_wall_dist = np.min(dist_to_wall)
    
    # Radius is half the minimum of the two distances
    r = min(min_pair_dist, min_wall_dist) / 2.0
    return r

def neg_radius(points_flat):
    points = points_flat.reshape(N, 2)
    return -compute_radius(points)

def hex_init(n_circles):
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate spacing
    # Area per circle ~ 1/n_circles
    # Hex cell area ~ s^2 * sqrt(3)/2
    # s^2 * sqrt(3)/2 * n_circles = 1
    # s = sqrt(2 / (n_circles * sqrt(3)))
    s = np.sqrt(2.0 / (n_circles * np.sqrt(3.0)))
    
    # Generate grid points
    # We need enough points to pick n_circles from
    # Let's generate a 10x10 grid of hex points
    rows = 10
    cols = 10
    points = []
    for i in range(rows):
        for j in range(cols):
            x = j * s + (0.5 * s if i % 2 == 1 else 0)
            y = i * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    points[:, 0] -= cx
    points[:, 1] -= cy
    
    # Scale to fit in square roughly
    # Max extent
    max_x = np.max(np.abs(points[:, 0]))
    max_y = np.max(np.abs(points[:, 1]))
    scale = 0.45 / max(max_x, max_y)
    points *= scale
    
    # Randomly select n_circles
    idx = np.random.choice(len(points), n_circles, replace=False)
    return points[idx]

def random_init(n_circles):
    return np.random.uniform(-0.5, 0.5, size=(n_circles, 2))

def grid_init(n_circles):
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Shuffle and take first N
    np.random.shuffle(grid)
    return grid[:n_circles]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    best_radius = -1
    best_points = None
    
    # List of initializers
    initializers = [hex_init, random_init, grid_init]
    
    # We will run multiple optimizations
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Select initializer
        init_func = initializers[i % len(initializers)]
        
        # Add small random noise to break symmetry
        x0 = init_func(N)
        x0 += np.random.normal(0, 0.01, x0.shape)
        
        # Ensure points are inside
        x0 = np.clip(x0, -0.49, 0.49)
        
        x0_flat = x0.flatten()
        
        # Run Nelder-Mead
        # maxiter depends on time budget
        # Let's give it a reasonable maxiter
        res = minimize(
            neg_radius, 
            x0_flat, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        r = -res.fun
        if r > best_radius:
            best_radius = r
            best_points = res.x.reshape(N, 2)
            
        # Optional: Local refinement step
        # Sometimes Nelder-Mead stops early. We can do a few more steps.
        # But for now, keep it simple.
        
    # Final check
    if best_points is None:
        # Fallback to grid
        return grid_init(N)
        
    # Ensure finite and within bounds
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will place it inside the block.
# EVOLVE-BLOCK-END
