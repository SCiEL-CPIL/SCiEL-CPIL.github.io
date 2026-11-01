"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_radius(p):
    """Compute the common radius for a set of centres inside [-0.5, 0.5]^3."""
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise distance constraint
    # N=50, so 50x50 matrix is small.
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)


def _generate_initial_positions():
    """Generate a good initial configuration using an FCC-like cluster scaled to fit."""
    # Generate FCC points in a large grid
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                # FCC basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k])
                points.append([i+0.5, j, k+0.5])
                points.append([i, j+0.5, k+0.5])
                
    points = np.array(points)
    
    # Center the cluster at the origin
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin to get the most compact cluster
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    # Take the N closest points
    pts = points[idx[:N]].copy()
    
    # Scale to fit inside the cube with some margin
    # We want to maximize the minimum distance to the wall.
    # Let's scale such that the max coordinate is around 0.4
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.4 / max_coord
        pts *= scale
        
    return pts


def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s limit
    
    # 1. Initialize with FCC-like cluster
    best_pts = _generate_initial_positions()
    best_r = _calc_radius(best_pts)
    
    # 2. Local Search via "Push" simulation
    # We iteratively try to increase the target radius.
    # If we can find a configuration where all spheres fit with radius >= target_r,
    # we accept it and increase target_r.
    
    # Start with a small increment
    dr = 0.0005
    current_r = best_r
    
    # We will keep trying to increase r until we fail or run out of time
    while time.time() - start_time < time_limit:
        target_r = current_r + dr
        
        # If target_r is unreasonably large, break
        if target_r > 0.15:
            break
            
        p = best_pts.copy()
        
        # Run a few iterations of force-directed relaxation to fit target_r
        # Success if min radius >= target_r
        success = False
        
        # Number of iterations per attempt
        # Keep it small to allow many attempts
        max_iter = 100
        
        for _ in range(max_iter):
            forces = np.zeros_like(p)
            
            # Boundary forces
            limits = 0.5 - target_r
            for axis in range(3):
                over_pos = p[:, axis] > limits
                over_neg = p[:, axis] < -limits
                
                # Push back with a strong force
                forces[over_pos, axis] -= 10.0
                forces[over_neg, axis] += 10.0
            
            # Pairwise forces
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            
            # Mask where dist < 2*target_r (overlap)
            mask = dists < 2 * target_r
            np.fill_diagonal(mask, False)
            
            if np.any(mask):
                # Calculate repulsive forces
                # F_i = sum_{j} (p_i - p_j) / |p_i - p_j| * strength
                # Only for pairs where dist < 2*target_r
                
                # Avoid division by zero
                safe_dists = np.where(dists > 1e-9, dists, 1.0)
                unit_vecs = diffs / safe_dists[:, :, np.newaxis]
                
                # Apply mask
                unit_vecs = unit_vecs * mask[:, :, np.newaxis]
                
                # Sum over neighbors (axis 1)
                pair_forces = np.sum(unit_vecs, axis=1)
                
                # Scale force
                forces += pair_forces * 10.0
                
            # Update positions
            # Use a small step size to avoid instability
            step = 0.01
            p += step * forces
            
            # Clamp to cube to prevent going outside
            p = np.clip(p, -0.5, 0.5)
            
            # Check if feasible
            curr_r = _calc_radius(p)
            if curr_r >= target_r:
                success = True
                break
        
        if success:
            # Accept the new configuration
            best_pts = p
            best_r = curr_r
            current_r = target_r
            # Increase dr slightly to speed up convergence
            # But be careful not to overshoot
            dr = dr * 1.05
        else:
            # Failed to increase r
            # Reduce dr to try smaller steps
            dr = dr * 0.9
            if dr < 1e-5:
                break
                
    # Final check and return
    final_r = _calc_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        
    return best_pts
# EVOLVE-BLOCK-END
