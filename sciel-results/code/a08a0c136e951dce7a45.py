import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Repulsion Relaxation (Thomson Problem)
    # Minimize E = sum(1 / dist^2)
    lr = 1e-4
    
    for step in range(2000):
        if time.time() > deadline - 60.0: # Leave 60s for local search
            break
            
        # Compute pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Gradient: sum_j 2 * (p_i - p_j) / ||p_i - p_j||^4
        weights = 2.0 * (inv_dist_sq ** 2)
        grad = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Project gradient to tangent space
        dot_prod = np.sum(grad * points, axis=1, keepdims=True)
        grad_tangent = grad - dot_prod * points
        
        # Update
        points = points + lr * grad_tangent
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    # 3. Local Search to Minimize Covering Radius
    # Use a dense grid to approximate the covering radius
    
    N_grid = 10000 # Reduced grid size for speed, still dense enough
    i_g = np.arange(N_grid) + 0.5
    z_g = 1.0 - 2.0 * i_g / N_grid
    phi_g = np.pi * (1.0 + np.sqrt(5.0)) * i_g
    r_g = np.sqrt(1.0 - z_g * z_g)
    grid_points = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
    
    def compute_covering_radius(pts):
        # Distance from each grid point to each config point
        diff = grid_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        max_min_dist_sq = np.max(min_dist_sq)
        # cos(theta) = 1 - dist_sq/2
        cos_theta = 1.0 - max_min_dist_sq / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
        
    current_cr = compute_covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.005
    cooling = 0.99
    
    steps = 0
    max_steps = 3000 # Reduced steps to ensure time limit
    
    while steps < max_steps and time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a random direction on the sphere
        v = np.random.normal(size=3)
        v = v / np.linalg.norm(v)
        
        # Project to tangent space at points[idx]
        dot = np.dot(v, points[idx])
        v_tangent = v - dot * points[idx]
        norm_v = np.linalg.norm(v_tangent)
        if norm_v < 1e-6:
            continue
        v_tangent = v_tangent / norm_v
        
        # Try a move
        step_size = 0.01
        new_point = points[idx] + step_size * v_tangent
        new_point = new_point / np.linalg.norm(new_point)
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        new_cr = compute_covering_radius(points)
        
        # Accept if better, or with probability exp(-(new_cr - current_cr)/T)
        if new_cr < current_cr or np.random.random() < np.exp(-(new_cr - current_cr) / T):
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = points.copy()
        else:
            points[idx] = old_point
            
        T = max(T_min, T * cooling)
        steps += 1
        
    return best_points
# EVOLVE-BLOCK-END
