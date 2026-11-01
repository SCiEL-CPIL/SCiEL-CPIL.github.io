import numpy as np
import time

N = 130

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # 10 seconds margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Dense grid for evaluation
    # Use a icosahedron subdivision or random points?
    # Random points are easier to generate densely.
    n_grid = 20000
    # Generate random points on sphere
    # Use rejection sampling or normal distribution
    grid = np.random.randn(n_grid, 3)
    grid /= np.linalg.norm(grid, axis=1, keepdims=True)
    
    # Optimization parameters
    beta = 10.0
    learning_rate = 0.05
    max_iter = 200
    
    best_points = points.copy()
    best_score = np.inf
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute distances from grid to all points
        # dist[y, i] = arccos(grid[y] . points[i])
        # Use dot product for speed, then arccos
        # Clip to [-1, 1] to avoid numerical issues
        dots = np.clip(grid @ points.T, -1.0, 1.0)
        dists = np.arccos(dots) # shape (n_grid, N)
        
        # Find min distance for each grid point
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        # Compute weights
        # Soft-max weights
        # exp(beta * min_dists)
        # Subtract max for stability
        max_d = np.max(min_dists)
        exp_dists = np.exp(beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        # Compute gradient for each point
        # grad[i] = sum_{y: min_indices[y] == i} weights[y] * grad_dist(y, points[i])
        grads = np.zeros_like(points)
        
        for i in range(N):
            mask = (min_indices == i)
            if not np.any(mask):
                continue
            w = weights[mask]
            g = grid[mask]
            p = points[i]
            
            # grad_dist = (g - (g.p)p) / sqrt(1 - (g.p)^2)
            dots_g = np.sum(g * p, axis=1)
            # Avoid division by zero
            denom = np.sqrt(np.maximum(1.0 - dots_g**2, 1e-12))
            grad_vec = (g - np.outer(dots_g, p)) / np.outer(denom, np.ones(3))
            
            # Weighted sum
            grads[i] = np.sum(w[:, np.newaxis] * grad_vec, axis=0)
        
        # Update points
        points -= learning_rate * grads
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Evaluate current score (max min dist)
        current_score = np.max(min_dists) * 180.0 / np.pi
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Decay learning rate
        learning_rate *= 0.99
        
        # Increase beta to focus on worst points
        beta *= 1.05
        
    return best_points
