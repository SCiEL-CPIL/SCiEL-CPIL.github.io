import numpy as np
import time

N = 100

def calc_radius(pos):
    """Calculate the score (max common radius) for a given set of centers."""
    if not np.all(np.isfinite(pos)):
        return 0.0
        
    # 1. Min distance to boundary
    # pos is in [-0.5, 0.5]
    # dist to wall = 0.5 - |pos|
    min_boundary = np.min(0.5 - np.abs(pos))
    
    # 2. Min distance between pairs
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_boundary, min_pair_dist / 2.0)
    return r

def optimize_packing(pos, r_target, iterations=1500, lr_start=0.05, lr_end=0.001, k_rep=1.0, k_wall=1.0):
    """Optimize positions to fit spheres of radius r_target."""
    pos = pos.copy()
    N = len(pos)
    
    for it in range(iterations):
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        
        overlaps = np.maximum(0, 2.0 * r_target - dists)
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        
        F = k_rep * overlaps * diffs / dists_safe[:, :, np.newaxis]
        net_pair_forces = np.sum(F, axis=1)
        
        # Wall repulsion
        lower_bound = -0.5 + r_target
        upper_bound = 0.5 - r_target
        
        wall_forces = np.zeros_like(pos)
        
        # X axis
        x_violation_lower = np.maximum(0, lower_bound - pos[:, 0])
        x_violation_upper = np.maximum(0, pos[:, 0] - upper_bound)
        wall_forces[:, 0] += k_wall * (x_violation_lower - x_violation_upper)
        
        # Y axis
        y_violation_lower = np.maximum(0, lower_bound - pos[:, 1])
        y_violation_upper = np.maximum(0, pos[:, 1] - upper_bound)
        wall_forces[:, 1] += k_wall * (y_violation_lower - y_violation_upper)
        
        # Z axis
        z_violation_lower = np.maximum(0, lower_bound - pos[:, 2])
        z_violation_upper = np.maximum(0, pos[:, 2] - upper_bound)
        wall_forces[:, 2] += k_wall * (z_violation_lower - z_violation_upper)
        
        total_forces = net_pair_forces + wall_forces
        
        # Learning rate schedule
        t = it / iterations
        lr = lr_start * (lr_end / lr_start) ** t
        
        pos += lr * total_forces
        
    return pos

def construct():
    t_start = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial Grid 5x5x4
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    
    best_pos = grid
    best_r = calc_radius(best_pos)
    
    rng = np.random.default_rng(42)
    
    # r targets to try
    # Start slightly above current best
    r_targets = np.arange(best_r + 0.002, 0.13, 0.005)
    
    for restart in range(10):
        if time.time() - t_start > deadline:
            break
            
        # Perturb initial grid
        noise_level = 0.02 if restart > 0 else 0.0
        pos = best_pos + rng.normal(0, noise_level, size=best_pos.shape)
        
        # Try to optimize for each r target
        for r_t in r_targets:
            if time.time() - t_start > deadline:
                break
                
            # If r_t is already less than best_r, skip
            if r_t <= best_r:
                continue
                
            pos_opt = optimize_packing(pos, r_t, iterations=1500)
            r_actual = calc_radius(pos_opt)
            
            if r_actual > best_r:
                best_r = r_actual
                best_pos = pos_opt
                
    return best_pos
