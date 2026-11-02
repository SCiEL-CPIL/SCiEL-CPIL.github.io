"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_centres = None
    
    # Precompute indices for pairs
    # This is expensive for N=200 (19900 pairs), but we can vectorize the objective
    # or just compute in the loop.
    
    def objective(params):
        """
        Minimize the sum of exponential penalties for small distances.
        params: flattened array of shape (2*N,)
        """
        x = params.reshape(N, 2)
        
        # Boundary distances
        # Distance to left, right, bottom, top
        # x[:, 0] is x-coord, x[:, 1] is y-coord
        # Square is [-0.5, 0.5]
        d_left = 0.5 - x[:, 0]
        d_right = 0.5 + x[:, 0]
        d_bottom = 0.5 - x[:, 1]
        d_top = 0.5 + x[:, 1]
        
        # Min distance to boundary for each circle
        # Note: x is constrained by L-BFGS-B bounds, so these should be positive
        # but we take min just in case of numerical drift
        d_bound = np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1)
        
        # Pairwise distances
        # Use broadcasting
        # x_i - x_j
        # To save memory and time, we can compute the full distance matrix
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This creates a 200x200 matrix, which is fine.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        np.fill_diagonal(dists, 1e9)
        
        # Flatten pairwise distances
        # We only need upper triangle to avoid double counting, but sum is symmetric
        # Let's just sum all i != j
        # To be efficient, we can take the sum of the upper triangle
        # Or just sum all and divide by 2? No, the penalty is per pair.
        # Summing all i!=j is fine.
        
        # Penalty for pairs
        # lambda = 200
        lambda_val = 200.0
        # To avoid overflow/underflow issues, we just sum exp(-lambda * dist)
        # If dist is small, exp is large.
        pair_penalties = np.exp(-lambda_val * dists)
        pair_sum = np.sum(pair_penalties)
        
        # Penalty for boundaries
        bound_penalties = np.exp(-lambda_val * d_bound)
        bound_sum = np.sum(bound_penalties)
        
        return pair_sum + bound_sum

    def compute_true_score(centres):
        """
        Compute the actual score: min(min pairwise dist, min boundary dist) / 2
        Wait, the score is r.
        r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (1/2 - |c_ik|) )
        """
        if centres is None:
            return -1.0
        
        # Boundary distance
        # min(0.5 - x, 0.5 + x) for each coord
        d_x = np.min(0.5 - np.abs(centres[:, 0]), axis=0) if centres.size else 0
        # Actually:
        d_left = 0.5 - centres[:, 0]
        d_right = 0.5 + centres[:, 0]
        d_bottom = 0.5 - centres[:, 1]
        d_top = 0.5 + centres[:, 1]
        
        min_bound = np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1))
        min_bound = np.min(min_bound)
        
        # Pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e9)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_bound)
        return r

    # Initialization strategy: Jittered Grid
    # Use a 15x15 grid (225 points) and pick 200
    # Or 14x14 (196) + 4 random?
    # 15x15 is denser, allows more freedom.
    M = 15
    grid_x = np.linspace(-0.45, 0.45, M) # Keep slightly inside to allow movement
    grid_y = np.linspace(-0.45, 0.45, M)
    
    # Create meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    # We have M*M = 225 points. We need 200.
    # Randomly select 200 indices
    indices = np.random.choice(M*M, N, replace=False)
    init_points = points[indices]
    
    # Add small jitter
    init_points += np.random.normal(0, 0.01, init_points.shape)
    
    # Clip to bounds (strictly inside)
    init_points = np.clip(init_points, -0.49, 0.49)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization loop
    while time.time() < deadline:
        # Try to optimize
        try:
            res = minimize(objective, init_points.ravel(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            opt_centres = res.x.reshape(N, 2)
            
            # Ensure points are within bounds (L-BFGS-B should handle, but check)
            opt_centres = np.clip(opt_centres, -0.5, 0.5)
            
            score = compute_true_score(opt_centres)
            
            if score > best_score:
                best_score = score
                best_centres = opt_centres.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
            
        # Generate a new initialization for the next restart
        # Re-jitter the grid
        init_points = points[indices] + np.random.normal(0, 0.02, (N, 2))
        init_points = np.clip(init_points, -0.49, 0.49)
        
        # Occasionally, try a different subset of the grid
        if np.random.rand() < 0.2:
             indices = np.random.choice(M*M, N, replace=False)
             init_points = points[indices] + np.random.normal(0, 0.01, (N, 2))
             init_points = np.clip(init_points, -0.49, 0.49)

    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        
    return best_centres
# EVOLVE-BLOCK-END
