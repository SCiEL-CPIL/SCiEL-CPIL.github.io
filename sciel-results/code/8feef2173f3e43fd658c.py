"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct centers of 100 equal spheres in [-0.5, 0.5]^3 to maximize the common radius.
    Uses a two-stage approach:
    1. A grid-based initial guess with random perturbation.
    2. Local optimization using L-BFGS-B on a smooth surrogate for the radius objective.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # --- Initial Guess: Grid with random perturbation ---
    # A 5x5x4 grid gives 100 points. This is a natural packing for 100 points.
    m = 5
    n_z = 4
    # Create grid points
    x = np.linspace(-0.4, 0.4, m)
    y = np.linspace(-0.4, 0.4, m)
    z = np.linspace(-0.45, 0.45, n_z)
    
    # Meshgrid and reshape to (N, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Ensure we have exactly N points
    if centers.shape[0] > N:
        centers = centers[:N]
    elif centers.shape[0] < N:
        # Pad with random points if needed (unlikely with 5x5x4=100)
        extra = np.random.uniform(-0.4, 0.4, size=(N - centers.shape[0], 3))
        centers = np.vstack([centers, extra])

    # Add small random perturbation to break symmetry and help optimization
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.5, 0.5)

    # --- Optimization ---
    def radius_obj(c):
        """
        Compute the radius for a given configuration.
        Returns negative radius to allow minimization.
        """
        # Pairwise distances
        # Use broadcasting to compute all pairwise distances
        # c has shape (N, 3)
        # dist_sq[i, j] = |c[i] - c[j]|^2
        dist_sq = np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=-1)
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Distance to walls
        # c_ik is the k-th coordinate of the i-th center
        # min_k (0.5 - |c_ik|) is the distance to the nearest wall
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return -r

    # Use L-BFGS-B for optimization
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Flatten centers to a 1D array for optimization
    x0 = centers.flatten()
    
    # Objective function for 1D input
    def obj_1d(x):
        c = x.reshape(N, 3)
        return radius_obj(c)
    
    # Run optimization
    try:
        res = minimize(obj_1d, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
        
        # Check time
        if time.time() > deadline:
            # Return best found so far (initial guess if optimization didn't finish)
            best_centers = centers
        else:
            best_centers = res.x.reshape(N, 3)
            
    except Exception:
        best_centers = centers

    # Final polish: Ensure points are strictly inside the cube
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_centers

# EVOLVE-BLOCK-END
