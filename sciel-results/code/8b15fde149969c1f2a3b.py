import numpy as np
import time
from scipy.spatial import cKDTree

N = 130

# EVOLVE-BLOCK-START

def _generate_grid(n_points):
    """Generate a reasonably uniform grid of points on the sphere."""
    # Fibonacci spiral is good for quasi-random distribution
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(points, grid):
    """Compute the maximum angular distance from grid points to nearest config point."""
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    # dists are Euclidean distances. Angular dist = 2 * arcsin(d/2)
    # For small angles, approx d. But we want degrees.
    # Actually, angular distance theta satisfies chord length d = 2 sin(theta/2).
    # theta = 2 arcsin(d/2).
    max_d = np.max(dists)
    max_theta = 2 * np.arcsin(max_d / 2.0)
    return np.degrees(max_theta)

def _optimize(points, grid, beta, steps, lr, time_limit):
    """
    Perform projected gradient descent on the sphere to minimize the soft-max of squared distances.
    """
    points = points.copy()
    n_grid = len(grid)
    
    # Precompute grid properties? No, just use it.
    # To speed up NN queries, we can use a KDTree rebuilt each step? 
    # Rebuilding a KDTree for 130 points is fast.
    
    for step in range(steps):
        if time.time() > time_limit:
            break
            
        # Compute distances from grid to points
        # Using broadcasting for small N and moderate grid size might be faster than KDTree if grid is small?
        # N=130, Grid=5000. 130*5000 = 650k distances. Vectorized dot product is fast.
        
        # dist_sq = 2 - 2 * (grid @ points.T)
        # Clip for numerical stability
        dot = grid @ points.T
        dot = np.clip(dot, -1.0, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        
        # Find nearest neighbor for each grid point
        nn_idx = np.argmin(dist_sq, axis=1)
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Soft max weights
        # L = 1/beta * log(sum(exp(beta * dist_sq)))
        # We want gradient of L
        # p_g = exp(beta * min_dist_sq_g) / sum(exp(beta * min_dist_sq_h))
        
        # To avoid overflow, subtract max
        max_val = np.max(min_dist_sq)
        exp_vals = np.exp(beta * (min_dist_sq - max_val))
        sum_exp = np.sum(exp_vals)
        p = exp_vals / sum_exp
        
        # Gradient calculation
        # grad_i = 2 * sum_{g: nn(g)=i} p_g * (x_i - g)
        # We can accumulate this using bincount or scatter
        # Vector x_i - g is (N, 3) if we broadcast?
        # g is (G, 3), x_i is (N, 3).
        # We need for each g, the vector (x_{nn(g)} - g).
        
        # Gather nearest points
        nn_points = points[nn_idx] # (G, 3)
        diffs = nn_points - grid # (G, 3)
        
        # Weighted sum for each point i
        # grad[i] = 2 * sum_{g: nn(g)=i} p_g * diffs[g]
        
        # Use np.add.at or bincount
        # We can flatten and use bincount
        # But diffs is 3D.
        # grad = np.zeros((N, 3))
        # np.add.at(grad, nn_idx, p[:, np.newaxis] * diffs * 2.0)
        
        # This is efficient enough for G=5000, N=130.
        
        grad = np.zeros((N, 3))
        np.add.at(grad, nn_idx, p[:, np.newaxis] * diffs * 2.0)
        
        # Update points
        points -= lr * grad
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_score = float('inf')
    best_points = None
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib)
    
    # 2. Random points (few seeds)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    # 3. Spherical coordinates grid (approx)
    # Just use random for now, Fibonacci is usually better start.
    
    # Optimization parameters
    # Coarse to fine grid
    grid_sizes = [2000, 5000, 10000]
    betas = [5.0, 10.0, 20.0]
    lrs = [0.1, 0.05, 0.01]
    steps_per_grid = [100, 200, 500]
    
    for config in configs:
        if time.time() > deadline - 20:
            break
            
        current_points = config.copy()
        
        for k, g_size in enumerate(grid_sizes):
            if time.time() > deadline - 5:
                break
                
            grid = _generate_grid(g_size)
            beta = betas[k]
            lr = lrs[k]
            steps = steps_per_grid[k]
            
            current_points = _optimize(current_points, grid, beta, steps, lr, deadline)
            
            # Evaluate
            score = _compute_covering_radius(current_points, grid)
            
            if score < best_score:
                best_score = score
                best_points = current_points.copy()
                
    # Final polish with high resolution grid and low LR
    if time.time() < deadline - 10:
        grid_fine = _generate_grid(20000)
        # Use a small step size
        best_points = _optimize(best_points, grid_fine, 20.0, 200, 0.005, deadline)
        final_score = _compute_covering_radius(best_points, grid_fine)
        if final_score < best_score:
            best_score = final_score
            
    if best_points is None:
        return fib
        
    return best_points

# EVOLVE-BLOCK-END
