"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the maximum radius for a given configuration.
    """
    n = len(centers)
    # Wall distances
    # dist to boundary is 0.5 - |coord|
    wall_dists = np.min(np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Vectorized calculation
    # diff (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Ignore self
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _hex_grid_init(n_points):
    """
    Generate a hexagonal grid of n_points.
    """
    # Estimate grid size
    rows = int(np.ceil(np.sqrt(n_points * 2 / np.sqrt(3))))
    cols = int(np.ceil(np.sqrt(n_points * np.sqrt(3) / 2)))
    
    # We want to center the grid in [-0.5, 0.5]
    # Let's generate a large enough grid and then crop/center
    # Hex spacing
    # Let's assume a target radius r ~ 0.07
    # d = 2*r = 0.14
    # Let's just generate points with d=1.0 and then scale/center?
    # Better: Generate coordinates, then center them.
    
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            pts.append([x, y])
            
    pts = np.array(pts)
    # Center the points
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Scale to fit in [-0.5, 0.5] roughly
    # Find the extent
    min_coords = np.min(pts, axis=0)
    max_coords = np.max(pts, axis=0)
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]
    
    # Scale factor to fit in 1.0 (with some margin for radius)
    # We want the bounding box to be < 1.0
    scale = 0.9 / max(width, height)
    pts *= scale
    
    # Take first n_points
    if len(pts) > n_points:
        # We should take the most central ones
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)[:n_points]
        pts = pts[indices]
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Initialize
    centers = _hex_grid_init(N)
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Simulated Annealing parameters
    T = 0.01
    T_min = 1e-6
    alpha = 0.9999  # Cooling rate
    step_size = 0.05
    
    # Precompute indices for pairs? Not strictly necessary with vectorization
    # But we can speed up by only checking neighbors? 
    # For N=50, full check is fine.
    
    step = 0
    while time.time() < deadline:
        # Perturb one random circle
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Random perturbation
        # Gaussian noise
        noise = np.random.normal(0, step_size, size=2)
        new_pos = old_pos + noise
        
        # Keep inside square? 
        # If we clamp, we might bias the distribution.
        # If we allow it, the wall penalty will handle it.
        # But the objective function handles wall distance.
        # So we don't strictly need to clamp, but it helps keep values reasonable.
        # Let's clamp to [-0.5, 0.5] to avoid huge wall violations
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        centers[i] = new_pos
        
        new_radius = _compute_radius(centers)
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.random() < np.exp(-delta / T):
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Revert
            centers[i] = old_pos
        
        # Cooling
        T *= alpha
        
        # Adaptive step size?
        # If we are stuck (no improvement for a long time), we might increase T or step size
        # For simplicity, let's keep it simple. The cooling schedule should handle exploration.
        
        step += 1
        
        # Check time periodically
        if step % 1000 == 0:
            if time.time() > deadline - 15.0:
                break
                
    return best_centers
# EVOLVE-BLOCK-END
