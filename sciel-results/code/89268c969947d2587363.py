"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def construct():
    # 1. Initialization: Grid-based
    # Create a 15x14 grid (210 points) in [-0.5, 0.5]^2
    rows = 14
    cols = 15
    x_coords = np.linspace(-0.5, 0.5, cols)
    y_coords = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Remove 10 points randomly to break symmetry and adjust count to 200
    # Use a fixed seed for reproducibility of the initial state, 
    # but the SA will vary if we don't seed the SA steps? 
    # Let's seed the removal for consistency in init, 
    # but let SA be random.
    np.random.seed(123)
    idx_to_remove = np.random.choice(210, 10, replace=False)
    points = np.delete(points, idx_to_remove, axis=0)
    
    # 2. Simulated Annealing
    def calc_radius(pts):
        # Boundary distance
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # Pairwise distances
        d = pdist(pts)
        min_pair = np.min(d) / 2.0
        
        return min(min_boundary, min_pair)
    
    r = calc_radius(points)
    
    # SA Parameters
    T = 0.01
    T_min = 1e-7
    alpha = 0.9995
    max_iter = 8000
    step_size = 0.01
    
    for i in range(max_iter):
        # Pick a random point
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        
        # Perturb with Gaussian noise
        points[idx] += np.random.normal(0, step_size, 2)
        
        # Calculate new radius
        new_r = calc_radius(points)
        
        # Accept/Reject
        if new_r > r:
            r = new_r
        else:
            # Boltzmann probability
            # Avoid overflow in exp
            delta = new_r - r
            if delta / T > -50: # exp(-50) is ~2e-22
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if np.random.rand() < prob:
                r = new_r
            else:
                points[idx] = old_point
        
        T *= alpha
        if T < T_min:
            break
            
    # Final check: ensure points are strictly inside?
    # The objective penalizes going outside, so they should be.
    # But just in case of numerical drift, clip slightly.
    # However, clipping might reduce r if they are very close to boundary.
    # If they are outside, r is negative.
    # If they are on boundary, r is determined by pairs.
    # Clipping to [-0.5, 0.5] is safe.
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END
