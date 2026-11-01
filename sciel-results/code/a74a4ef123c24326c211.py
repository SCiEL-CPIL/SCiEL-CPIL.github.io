import numpy as np

N = 300

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing with Optimized Min Tracking
    rng = np.random.default_rng(42)
    
    # 1. Initialization
    r_init = np.sqrt(rng.random(N))
    theta_init = rng.random(N) * 2 * np.pi
    points = np.column_stack((r_init * np.cos(theta_init), r_init * np.sin(theta_init)))
    
    # Initialize distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Initial min tracking
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i_min, j_min = min_pair_idx
    
    min_boundary_val = np.min(dists_boundary)
    min_boundary_idx = np.argmin(dists_boundary)
    
    current_r = min(min_pair_val / 2.0, min_boundary_val)
    
    # SA Parameters
    T_start = 0.05
    T_end = 1e-6
    iterations = 500000
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    T = T_start
    
    for it in range(iterations):
        idx = rng.integers(0, N)
        
        # Move
        step_mag = T
        angle = rng.random() * 2 * np.pi
        move = np.array([step_mag * np.cos(angle), step_mag * np.sin(angle)])
        
        old_point = points[idx].copy()
        new_point = old_point + move
        points[idx] = new_point
        
        # Update distances for idx
        diff_row = new_point - points
        new_dists_row = np.linalg.norm(diff_row, axis=1)
        new_dists_row[idx] = np.inf
        
        dists[idx, :] = new_dists_row
        dists[:, idx] = new_dists_row
        
        # Update boundary
        new_boundary = 1.0 - np.linalg.norm(new_point)
        dists_boundary[idx] = new_boundary
        
        # Update min tracking
        # 1. Check if idx was part of the old min pair
        idx_was_min_pair = (idx == i_min) or (idx == j_min)
        
        if idx_was_min_pair:
            # Need to find new global min pair
            min_pair_val = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i_min, j_min = min_pair_idx
        else:
            # Old min pair is still valid.
            # Check if new distances involving idx are smaller
            min_dist_involving_idx = np.min(new_dists_row)
            if min_dist_involving_idx < min_pair_val:
                # New global min involves idx
                # Find which j gives this min
                j_new = np.argmin(new_dists_row)
                min_pair_val = min_dist_involving_idx
                # Ensure i < j for consistency? Not strictly necessary for value, but for indices
                if idx < j_new:
                    i_min, j_min = idx, j_new
                else:
                    i_min, j_min = j_new, idx
        
        # 2. Check boundary
        # Similar logic for boundary min
        idx_was_min_boundary = (idx == min_boundary_idx)
        
        if idx_was_min_boundary:
            min_boundary_val = np.min(dists_boundary)
            min_boundary_idx = np.argmin(dists_boundary)
        else:
            if new_boundary < min_boundary_val:
                min_boundary_val = new_boundary
                min_boundary_idx = idx
        
        # Calculate new radius
        new_r = min(min_pair_val / 2.0, min_boundary_val)
        
        # Acceptance
        delta_r = new_r - current_r
        if delta_r > 0:
            current_r = new_r
        else:
            prob = np.exp(delta_r / T)
            if rng.random() < prob:
                current_r = new_r
            else:
                # Reject
                points[idx] = old_point
                # Restore distances
                diff_row_old = old_point - points
                old_dists_row = np.linalg.norm(diff_row_old, axis=1)
                old_dists_row[idx] = np.inf
                dists[idx, :] = old_dists_row
                dists[:, idx] = old_dists_row
                dists_boundary[idx] = 1.0 - np.linalg.norm(old_point)
                
                # Restore min tracking
                # This is tricky because we modified i_min, j_min, min_boundary_idx etc.
                # It is safer to just recalculate min tracking from scratch on reject?
                # Or store old values?
                # Given the low probability of reject (early on), and high probability of accept (early on),
                # and later on delta_r is small, 
                # Recalculating min tracking on reject is O(N^2) or O(N).
                # Let's just recalculate the min pair and min boundary on reject to be safe and simple.
                # Since reject happens less often as we cool, this is acceptable.
                min_pair_val = np.min(dists)
                min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_min, j_min = min_pair_idx
                min_boundary_val = np.min(dists_boundary)
                min_boundary_idx = np.argmin(dists_boundary)
        
        T *= cooling_rate
        
    return points
# EVOLVE-BLOCK-END
