import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Force-Directed Repulsion to maximize the minimum distance between circles.
    This approach applies repulsive forces to all "stressed" points (those close
    to neighbors or walls), allowing for global adjustments rather than just
    moving the single bottleneck pair.
    """
    
    def evaluate(points):
        """
        Returns the minimum radius r for the given configuration.
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        # 1. Distance to walls
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # 2. Distance between pairs
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def compute_forces(points, threshold, strength):
        """
        Computes net repulsive force on each point.
        Forces are active for pairs/walls closer than 'threshold'.
        Magnitude is proportional to (threshold - dist) * strength.
        """
        n = len(points)
        forces = np.zeros_like(points)
        
        # Pairwise forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Mask for active pairs (dist < threshold and i != j)
        mask = (dists < threshold) & (np.eye(n) == 0)
        
        # Safe division
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        
        # Weights: strength * (threshold - dist) for active pairs
        weights = np.where(mask, strength * (threshold - dists), 0.0)
        
        # Force vectors: unit vector from j to i
        force_vectors = diff / safe_dists[:, :, np.newaxis]
        
        # Sum contributions from all j to each i
        pair_forces = np.sum(force_vectors * weights[:, :, np.newaxis], axis=1)
        forces += pair_forces
        
        # Wall forces
        d_left = points[:, 0] + 0.5
        d_right = 0.5 - points[:, 0]
        d_bottom = points[:, 1] + 0.5
        d_top = 0.5 - points[:, 1]
        
        # Force from left wall (push right if close)
        mask_left = d_left < threshold
        forces[:, 0] += np.where(mask_left, strength * (threshold - d_left), 0.0)
        
        # Force from right wall (push left if close)
        mask_right = d_right < threshold
        forces[:, 0] -= np.where(mask_right, strength * (threshold - d_right), 0.0)
        
        # Force from bottom wall (push up if close)
        mask_bottom = d_bottom < threshold
        forces[:, 1] += np.where(mask_bottom, strength * (threshold - d_bottom), 0.0)
        
        # Force from top wall (push down if close)
        mask_top = d_top < threshold
        forces[:, 1] -= np.where(mask_top, strength * (threshold - d_top), 0.0)
        
        return forces

    # Initialization: Grid with jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    best_points = np.zeros((N, 2))
    best_score = 0.0
    
    start_time = time.time()
    time_limit = 110
    
    # Run multiple restarts if time permits
    restart_count = 0
    max_restarts = 10
    
    while time.time() - start_time < time_limit and restart_count < max_restarts:
        restart_count += 1
        # New initialization for each restart
        points = grid[:N].copy()
        points += np.random.normal(0, 0.015, size=points.shape)
        points = np.clip(points, -0.49, 0.49)
        
        current_points = points.copy()
        current_score = evaluate(current_points)
        
        # Parameters for this restart
        initial_threshold = 0.12
        final_threshold = 0.015
        strength = 1.0
        step_size = 0.02
        
        # Determine time allocation for this restart
        # Use remaining time, but cap it to allow for restarts if early
        remaining_time = time_limit - (time.time() - start_time)
        if restart_count == 1:
            alloc_time = min(remaining_time, 50) # First run gets more time
        else:
            alloc_time = min(remaining_time, 20) # Subsequent runs shorter
        
        restart_start = time.time()
        
        while time.time() - restart_start < alloc_time:
            # Adaptive threshold: decrease over time within this restart
            progress = min(1.0, (time.time() - restart_start) / alloc_time)
            threshold = initial_threshold * (1.0 - progress) + final_threshold * progress
            
            # Compute forces
            forces = compute_forces(current_points, threshold, strength)
            
            # Normalize forces to prevent huge steps
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_forces = forces / norms
            
            # Update
            current_points += step_size * normalized_forces
            
            # Clip
            current_points = np.clip(current_points, -0.5, 0.5)
            
            # Evaluate
            current_score = evaluate(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
            # Occasionally add noise to escape local minima
            if np.random.rand() < 0.02:
                current_points += np.random.normal(0, 0.005, size=current_points.shape)
                current_points = np.clip(current_points, -0.49, 0.49)
                
            # Decrease step size slowly
            step_size *= 0.998
            
            # If forces are very small, we might be at a local minimum
            # The noise injection handles escape.
            
    return best_points
# EVOLVE-BLOCK-END
