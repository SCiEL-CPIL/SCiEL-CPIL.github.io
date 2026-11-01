"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    # Use a BCC (Body-Centered Cubic) lattice initialization for better start
    # BCC is denser than FCC or SC.
    # Generate 4x4x4 BCC lattice (128 points)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                grid_pts.append([i, j, k])
                grid_pts.append([i+0.5, j+0.5, k+0.5])
    
    grid_pts = np.array(grid_pts)
    # Normalize to [0, 1]
    grid_pts = grid_pts / 3.5
    # Shift to [-0.5, 0.5]
    grid_pts = grid_pts - 0.5
    
    # We have 128 points. We need 100.
    # Keep the 100 points with the largest minimum distance to their nearest neighbor
    # This ensures we start with a relatively uniform distribution
    d = np.linalg.norm(grid_pts[:, np.newaxis, :] - grid_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    
    # Sort by min distance descending, take top 100
    sorted_indices = np.argsort(min_d)[::-1]
    initial_pts = grid_pts[sorted_indices[:N]].copy()
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    initial_pts += np.random.normal(0, 0.01, initial_pts.shape)
    
    # Clip to valid range to ensure validity (centers must be strictly inside)
    initial_pts = np.clip(initial_pts, -0.499, 0.499)
    
    # Function to compute the radius (score)
    def compute_radius(pts):
        # Min distance between any two points
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_inter = np.min(d) / 2.0
        
        # Min distance to walls
        # 0.5 - |x|
        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists)
        
        return min(min_inter, min_wall)
    
    current_pts = initial_pts
    current_r = compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Simulated Annealing Parameters
    T = 0.05
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        step_size = T * 0.5
        perturbation = np.random.normal(0, step_size, (N, 3))
        new_pts = current_pts + perturbation
        
        # Clip to box to ensure validity
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = compute_radius(new_pts)
        
        # Acceptance criterion
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        else:
            delta = new_r - current_r
            if delta < 0 and T > 1e-9:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_r = new_r
        
        # Cool down
        T *= cooling_rate
        
    return best_pts
# EVOLVE-BLOCK-END
