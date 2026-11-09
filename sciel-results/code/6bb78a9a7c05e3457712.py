"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 2)
    
    if centers.shape[0] != N:
        return 0.0

    # Distance to walls
    # centers[:, 0] are x coords, centers[:, 1] are y coords
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    # Min distance to wall for each circle
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    if N > 1:
        # Use scipy.spatial.distance.cdist if available, else manual
        # Since SciPy is available:
        from scipy.spatial.distance import cdist
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = np.inf
    
    return min(min_wall, min_pair)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initialize with a grid-like structure which is a good starting point
    # 7 columns, 8 rows approx
    # We want to maximize min distance.
    # A regular grid is a strong candidate.
    
    def init_grid(cols, rows):
        # Place circles in a grid
        # Adjust spacing to fit in [-0.5, 0.5]
        # We want to maximize r.
        # For a grid of C x R, the limiting factor is usually the spacing.
        # Let's just place them evenly and let SA optimize.
        xs = np.linspace(-0.4, 0.4, cols)
        ys = np.linspace(-0.4, 0.4, rows)
        xx, yy = np.meshgrid(xs, ys)
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        # If we have more than N, take first N
        # If we have less, we need to add more? 
        # 7*8=56. We take 50.
        if len(centers) > N:
            centers = centers[:N]
        return centers

    best_centers = init_grid(7, 8)
    best_radius = _compute_radius(best_centers)
    
    # Simulated Annealing
    T = 0.05
    step_size = 0.05
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    # Precompute neighbor list? No, just use cdist each time, N=50 is small.
    # cdist(50, 50) is very fast.
    
    iterations = 0
    while time.time() < deadline:
        # Pick a random circle to move
        idx = np.random.randint(0, N)
        
        # Generate a random move
        # Step size can be adaptive
        move = np.random.normal(0, step_size, size=2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += move
        
        # Clamp to square
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Always accept improvements
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
                # Reduce step size slightly on improvement?
                # Or keep it to explore.
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_centers = new_centers
                current_radius = new_radius
        
        # Cool down
        T *= 0.999
        # Also reduce step size slowly
        step_size *= 0.9995
        
        iterations += 1
        
        # Prevent T from becoming too small
        if T < 1e-8:
            T = 1e-8
            step_size = 0.001
            # Reset if stuck?
            # Simple restart if no improvement for a long time
            # But we have a deadline.
            
    return best_centers
# EVOLVE-BLOCK-END
