"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius.
    # Strategy: Start from a good initial guess (body-centered cubic-like or random restarts),
    # then optimize using scipy to maximize the minimum distance between centers and to walls.
    
    # The objective is to maximize r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|))
    # We minimize the negative of this, i.e., maximize the minimum of:
    #   d_ij/2 for all i<j
    #   0.5 - |x_i|, 0.5 - |y_i|, 0.5 - |z_i| for all i,k
    
    # Since scipy minimizes, we define a function that returns -min_radius.
    # But min_radius is a min of many terms, so we can use a soft approach or direct minimization.
    
    # A better approach: use a penalty-based or direct optimization where we try to maximize
    # the minimum pairwise distance and wall distance.
    
    # Let's define the objective as negative of the minimum radius.
    # To make it smooth, we can use a log-sum-exp approximation, but for simplicity,
    # let's use a direct approach with many random restarts and local optimization.
    
    def min_radius(positions):
        """Compute the minimum radius given positions."""
        positions = positions.reshape(-1, 3)
        # Pairwise distances
        from scipy.spatial.distance import pdist
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def objective(positions):
        """Negative of minimum radius to be minimized."""
        return -min_radius(positions)
    
    # Initial guess: use a body-centered cubic arrangement or random
    # Let's try a few different initializations
    
    best_positions = None
    best_radius = -1.0
    
    # Try multiple random restarts
    np.random.seed(42)
    
    for restart in range(5):
        if restart == 0:
            # Start with a simple grid
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            positions = grid[:N].copy()
        else:
            # Random positions within the cube
            positions = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Optimize
        res = minimize(objective, positions.flatten(), method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (N * 3),
                       options={'maxiter': 200, 'ftol': 1e-10})
        
        pos_opt = res.x.reshape(-1, 3)
        r_opt = min_radius(pos_opt)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_positions = pos_opt
    
    # Ensure positions are within bounds
    best_positions = np.clip(best_positions, -0.5, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
