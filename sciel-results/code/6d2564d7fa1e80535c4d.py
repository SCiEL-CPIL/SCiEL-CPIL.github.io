"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Pre-compute a grid on the sphere for evaluating covering radius
    # Use a Fibonacci lattice for evaluation points to ensure uniform distribution
    # This is more accurate than a lat-long grid for finding the true worst point
    n_eval = 5000
    i_eval = np.arange(n_eval) + 0.5
    z_eval = 1.0 - 2.0 * i_eval / n_eval
    phi_eval = np.pi * (1.0 + np.sqrt(5.0)) * i_eval
    r_eval = np.sqrt(1.0 - z_eval * z_eval)
    eval_points = np.column_stack([r_eval * np.cos(phi_eval), r_eval * np.sin(phi_eval), z_eval])
    eval_points = eval_points / np.linalg.norm(eval_points, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 115.0
    
    step_size = 0.05
    decay = 0.995
    repulsive_strength = 0.01
    patience = 0
    max_patience = 300
    prev_radius = np.pi
    best_points = points.copy()
    best_radius = np.pi
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate covering radius on the evaluation grid
        # Calculate dot products: (n_eval, N)
        dots = np.dot(eval_points, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Max dot product corresponds to min angular distance
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        
        current_radius = np.max(min_dists)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if current_radius < prev_radius:
            prev_radius = current_radius
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                # Reset to best and add noise to escape local minimum
                points = best_points + 0.05 * np.random.randn(N, 3)
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
                patience = 0
                step_size = max(0.01, step_size * 0.5)
        
        # Find the evaluation point with the largest distance (the "hole")
        worst_idx = np.argmax(min_dists)
        worst_point = eval_points[worst_idx]
        
        # Find the 3 nearest points to the worst_point
        dots_to_worst = np.dot(points, worst_point)
        # Sort indices by dot product descending (closest first)
        nearest_indices = np.argsort(-dots_to_worst)[:3]
        
        # Move these 3 points TOWARDS the worst_point
        # This is the key correction: move towards the hole to cover it
        for idx in nearest_indices:
            p = points[idx]
            # Vector from p to worst_point
            # Project onto tangent space of p
            # t = worst_point - (worst_point . p) p
            # This vector points from p towards worst_point (tangent component)
            proj = worst_point - np.dot(worst_point, p) * p
            t_norm = np.linalg.norm(proj)
            if t_norm > 1e-6:
                t = proj / t_norm
                # Move towards worst_point
                points[idx] = _normalize(p + step_size * t)
            else:
                # If p is very close to worst_point, move randomly or away to avoid stagnation?
                # Actually if p is at worst_point, radius is 0. But we are looking for max radius.
                # So worst_point is far from all points.
                pass
        
        # Apply global repulsive force to maintain uniformity and prevent clumping
        # Use a simplified O(N^2) loop but with early exit or vectorization if possible
        # For N=130, O(N^2) is 16900 ops, which is fast.
        forces = np.zeros((N, 3))
        for i in range(N):
            pi = points[i]
            for j in range(N):
                if i == j: continue
                pj = points[j]
                vec = pi - pj
                d = np.linalg.norm(vec)
                if d < 1e-6:
                    # Random jitter if coincident
                    vec = np.random.randn(3)
                    d = np.linalg.norm(vec)
                
                # Tangential component of repulsion
                # Force is along pi - pj, projected to tangent plane of pi
                # tang = vec - (vec . pi) pi
                tang = vec - np.dot(vec, pi) * pi
                norm_tang = np.linalg.norm(tang)
                if norm_tang > 1e-6:
                    # Magnitude decays with distance squared
                    mag = repulsive_strength / (d * d)
                    forces[i] += (tang / norm_tang) * mag
        
        # Apply forces
        points = points + forces
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= decay
        
    return best_points
# EVOLVE-BLOCK-END
