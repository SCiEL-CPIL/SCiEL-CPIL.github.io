import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    If invalid (points outside), radius is -1.
    """
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, radius is invalid (negative)
    if np.any(bound_dists < 0):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint using cdist for efficiency
    dists = cdist(positions, positions)
    
    # Ignore self-distances
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def soft_min_radius(positions, alpha=20.0):
    """
    Compute a smooth approximation of the minimum radius.
    This provides gradients for L-BFGS-B optimization.
    """
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, return penalty
    if np.any(bound_dists < 0):
        return -1.0 - np.max(-bound_dists)
    
    # Pairwise distances
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    
    # Soft minimum of pairwise distances / 2
    # Use log-sum-exp trick for numerical stability
    min_pair_dist = np.min(dists)
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    # For large alpha, this approaches min(d)
    scaled = -alpha * (dists - min_pair_dist)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    soft_min_pair = min_pair_dist - np.log(np.sum(np.exp(scaled))) / alpha
    
    # Soft min of boundary distances
    min_bound = np.min(bound_dists)
    scaled_bound = -alpha * (bound_dists - min_bound)
    scaled_bound = np.clip(scaled_bound, -500, 500)
    soft_min_bound = min_bound - np.log(np.sum(np.exp(scaled_bound))) / alpha
    
    # Combine: soft min of the two soft mins
    # Actually, we want min(soft_min_pair/2, soft_min_bound)
    # Let's just take the minimum of the two
    r = min(soft_min_pair / 2.0, soft_min_bound)
    
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
    
    # Add small noise to break symmetry
    r_init, _ = compute_radius(positions)
    noise_std = max(0.001, r_init * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius, _ = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. L-BFGS-B optimization on soft-min surrogate
    # Maximize radius = minimize -radius
    def objective(x):
        """Objective function for L-BFGS-B: minimize -soft_min_radius"""
        pos = x.reshape(N, 2)
        # Penalty for going outside
        max_abs = np.max(np.abs(pos), axis=1)
        bound_dists = 0.5 - max_abs
        if np.any(bound_dists < 0):
            return 1.0 + np.max(-bound_dists)
        r = soft_min_radius(pos, alpha=20.0)
        return -r
    
    # Run L-BFGS-B optimization
    x0 = positions.flatten()
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 2),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.success or result.fun < -best_radius:
            new_positions = result.x.reshape(N, 2)
            new_radius, is_valid = compute_radius(new_positions)
            if is_valid and new_radius > best_radius:
                best_radius = new_radius
                best_positions = new_positions.copy()
                
    except Exception:
        pass
    
    # 3. Simulated Annealing polish
    T = 0.001
    T_end = 1e-7
    alpha_cool = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    positions = best_positions.copy()
    current_radius = best_radius
    
    while time.time() < deadline:
        for _ in range(100):
            if time.time() >= deadline:
                break
                
            total_count += 1
            idx = np.random.randint(N)
            
            if total_count % 100 == 0:
                accept_rate = accept_count / max(1, total_count)
                if accept_rate < 0.2:
                    step_size *= 1.2
                elif accept_rate > 0.8:
                    step_size *= 0.8
                step_size = np.clip(step_size, min_step_size, max_step_size)
                accept_count = 0
                total_count = 0
                
            delta = np.random.normal(0, step_size, size=2)
            old_pos = positions[idx].copy()
            positions[idx] = old_pos + delta
            new_radius, is_valid = compute_radius(positions)
            
            if not is_valid:
                new_radius = -1.0
            
            if new_radius > current_radius:
                current_radius = new_radius
                accept_count += 1
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        accept_count += 1
                    else:
                        positions[idx] = old_pos
                else:
                    positions[idx] = old_pos
                    
        T *= alpha_cool
        if T < T_end:
            T = T_end
    
    return best_positions
