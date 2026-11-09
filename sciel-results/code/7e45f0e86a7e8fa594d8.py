import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # 1. Initialize points on a Fibonacci sphere for good initial spread
    # Fibonacci sphere algorithm
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    
    points = np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    
    # 2. Optimize to maximize minimum distance (Tammes problem approximation)
    # We use a simple gradient ascent on the sum of inverse squared distances
    # or simply repulsion forces.
    
    # Let's use a repulsion force model: F ~ 1/r^2
    # We will iterate for a fixed number of steps
    steps = 500
    learning_rate = 0.01
    
    for _ in range(steps):
        # Calculate pairwise distances and vectors
        # points shape (n, d)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists[dists == 0] = 1e-10
        
        # Force matrix: F_ij = (r_i - r_j) / dist_ij^2
        # We want to move points apart, so force on i from j is along (r_i - r_j)
        # Normalize the direction and scale by 1/dist^2
        # Actually, let's use the gradient of -sum(1/d^2) which is sum(2/d^3 * r_vec)
        # Let's just use simple repulsion: F = r_vec / dist^3 (Coulomb is 1/r^2 force, so F_vec = r_vec / r^3)
        
        # Compute forces
        # dists shape (n, n)
        # diff shape (n, n, d)
        
        # We want F_ij = diff_ij / dists_ij^3
        # But diff_ij = r_i - r_j
        # Force on i is sum_j F_ij
        
        # To avoid self-interaction and numerical issues
        np.fill_diagonal(dists, 1.0) # Placeholder, we will mask it
        
        # Calculate inverse distance cubed for scaling
        inv_dist_cubed = 1.0 / (dists ** 3)
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # Forces: (n, n, d) * (n, n, 1)
        forces = diff * inv_dist_cubed[:, :, np.newaxis]
        
        # Sum forces for each point
        net_forces = np.sum(forces, axis=1)
        
        # Update positions
        points += learning_rate * net_forces
        
        # Project back to the unit sphere to maintain d_max constraint implicitly
        # and to keep the configuration "spherical" which is optimal for min/max ratio
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return points
