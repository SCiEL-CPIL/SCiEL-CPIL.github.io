import numpy as np
import time

N = 100

def _compute_score_fast(centers):
    """
    Fast score computation using squared distances to avoid sqrt.
    Returns (radius, min_pair_sq_dist, min_wall_dist).
    """
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(sq_dists, np.inf)
    
    min_pair_sq_dist = np.min(sq_dists)
    min_pair_dist = np.sqrt(min_pair_sq_dist)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius, min_pair_sq_dist, min_wall_dist

def _get_active_points(centers, radius, tol=1e-5):
    """
    Identify points that are close to the limiting radius.
    Uses squared distances for speed.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Points involved in pairs with distance < 2*radius + tol
    threshold_sq = (2.0 * radius + tol) ** 2
    active_pairs = sq_dists < threshold_sq
    
    # Points close to walls
    wall_dists = 0.5 - np.abs(centers)
    active_walls = (wall_dists < (radius + tol)).any(axis=1)
    
    active_indices = np.unique(np.concatenate([
        np.where(active_pairs)[0],
        np.where(active_pairs)[1],
        np.where(active_walls)[0]
    ]))
    
    return active_indices

def _compute_score(centers):
    """
    Compute the maximum radius for the given centers.
    Returns just radius for compatibility.
    """
    radius, _, _ = _compute_score_fast(centers)
    return radius

def construct():
    # Initialize with 5x5x5 grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_radius, _, _ = _compute_score_fast(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Local Search Parameters
    step_size = 0.02
    kick_interval = 500  # Add noise every N iterations if no improvement
    
    iteration = 0
    no_improve_count = 0
    current_radius = best_radius
    
    while time.time() - start_time < time_limit:
        # Check time more frequently to avoid overshooting
        if (iteration % 10 == 0) and (time.time() - start_time > time_limit - 5.0):
            break
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck for a while, kick with larger perturbation
        if no_improve_count > kick_interval:
            # Gradually increase kick size based on how long we've been stuck
            kick_size = 0.05 + 0.05 * min(no_improve_count / 5000.0, 1.0)
            centers += np.random.normal(0, kick_size, size=(N, 3))
            # Clip to cube to ensure validity
            centers = np.clip(centers, -0.49, 0.49)
            no_improve_count = 0
            current_radius, _, _ = _compute_score_fast(centers)
            continue
            
        # Find active points
        active_indices = _get_active_points(centers, current_radius)
        
        if len(active_indices) == 0:
            # Should not happen if radius is correct, but handle just in case
            active_indices = np.random.choice(N, size=min(5, N), replace=False)
            
        # Try to move one active point
        idx = active_indices[np.random.randint(len(active_indices))]
        
        # Generate random direction
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        # Try multiple step sizes for better exploration
        step_sizes = [step_size, step_size * 0.5, step_size * 2.0]
        improved = False
        
        for ss in step_sizes:
            for sign in [1, -1]:
                new_centers = centers.copy()
                new_centers[idx] += sign * ss * direction
                
                # Ensure points stay inside the cube
                new_centers = np.clip(new_centers, -0.49, 0.49)
                
                new_radius, _, _ = _compute_score_fast(new_centers)
                
                if new_radius > current_radius:
                    centers = new_centers
                    current_radius = new_radius
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a small random kick on the point
            small_kick = 0.005
            for _ in range(3):
                new_centers = centers.copy()
                new_centers[idx] += np.random.normal(0, small_kick, size=3)
                new_centers = np.clip(new_centers, -0.49, 0.49)
                new_radius, _, _ = _compute_score_fast(new_centers)
                if new_radius > current_radius:
                    centers = new_centers
                    current_radius = new_radius
                    improved = True
                    break
        
        iteration += 1
        
        # Adaptive step size
        if improved:
            # Keep or slightly increase step size
            step_size = min(step_size * 1.05, 0.05)
        else:
            # Reduce step size
            step_size *= 0.98
            if step_size < 0.005:
                step_size = 0.01  # Reset if too small

    # Final check
    final_radius, _, _ = _compute_score_fast(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centers

# EVOLVE-BLOCK-END
