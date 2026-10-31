import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Initialize with a hexagonal grid
    # We want to fit N=26 circles.
    # A hex grid with spacing s has area per circle ~ s^2 * sqrt(3)/2.
    # Total area 1. N * s^2 * sqrt(3)/2 <= 1 => s <= sqrt(2 / (N * sqrt(3)))
    # For N=26, s ~ sqrt(2 / (26 * 1.732)) ~ sqrt(0.044) ~ 0.21
    # Let's start with a reasonable s.
    
    def hex_grid_positions(n, spacing):
        positions = []
        # Estimate grid size
        rows = int(np.sqrt(n * 2 / np.sqrt(3))) + 2
        cols = int(np.sqrt(n * 2 / np.sqrt(3))) + 2
        
        # Generate a larger grid and pick the best n
        # Or just generate enough
        y = 0.5 * spacing
        row_idx = 0
        while len(positions) < n + 10: # Generate a bit extra
            x_offset = (row_idx % 2) * (spacing * 0.5)
            x = x_offset + 0.5 * spacing
            col_idx = 0
            while len(positions) < n + 10:
                positions.append([x, y])
                x += spacing
                col_idx += 1
            y += spacing * np.sqrt(3) / 2
            row_idx += 1
        
        return np.array(positions[:n])

    # Initial spacing
    # If we use s=0.18, r=0.09. Sum=26*0.09 = 2.34.
    # Grid seed: side=ceil(sqrt(26))=6. r=0.5/6 - eps ~ 0.083. Sum ~ 2.16.
    # So hex grid should be better.
    
    best_score = 0
    best_positions = None
    
    # Try a few different initial spacings and orientations
    for trial in range(10):
        if time.time() > deadline:
            break
            
        # Randomize starting position slightly or spacing
        spacing = 0.18 + 0.01 * trial # Tune spacing
        pos = hex_grid_positions(N, spacing)
        
        # Center the grid roughly
        # Calculate bounding box
        min_x, min_y = pos.min(axis=0)
        max_x, max_y = pos.max(axis=0)
        w = max_x - min_x
        h = max_y - min_y
        
        # Scale and shift to fit in [0,1] with some margin
        # We want to maximize the minimum distance to walls.
        # If we scale to fit exactly, the margin is 0.
        # We should center it.
        
        # Actually, let's just optimize the positions directly.
        
        # Optimization Loop
        # We want to maximize r such that:
        # x_i, y_i, 1-x_i, 1-y_i >= r
        # ||p_i - p_j|| >= 2r
        
        # This is equivalent to maximizing the minimum of these values.
        # Let's use a simple iterative method:
        # 1. Estimate current max possible r (min clearance).
        # 2. Move points to increase the minimum clearance.
        
        curr_pos = pos.copy()
        
        # Normalize to be inside [0,1] initially
        # If the grid is too big, scale it down.
        # Let's just assume the grid fits and optimize.
        # If it doesn't fit, the min clearance will be negative or small.
        
        # To ensure it fits, let's scale the initial positions to fit in [0.1, 0.9]
        # Then expand.
        
        # Better: Start with a valid configuration.
        # Scale curr_pos to fit in [0, 1]
        min_x, min_y = curr_pos.min(axis=0)
        max_x, max_y = curr_pos.max(axis=0)
        w = max_x - min_x
        h = max_y - min_y
        scale = 0.8 / max(w, h) # Fit in 0.8x0.8 box to be safe
        curr_pos = (curr_pos - [min_x, min_y]) * scale
        curr_pos += [0.1, 0.1] # Center in [0.1, 0.9]
        
        # Now optimize
        # We will try to increase the minimum distance.
        # Gradient approximation:
        # For each point i, the gradient of the min distance is the direction away from the closest constraint.
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Calculate distances to walls and other circles
            # dist_to_walls[i] = min(x_i, 1-x_i, y_i, 1-y_i)
            # dist_to_circles[i, j] = 0.5 * ||p_i - p_j||
            
            # We want to maximize min(all these distances).
            # Let r_min be the current global min.
            # Identify points involved in the min constraints.
            
            # Efficiently:
            # 1. Find min wall distance for each point.
            wall_dists = np.minimum(np.minimum(curr_pos[:, 0], 1 - curr_pos[:, 0]), 
                                    np.minimum(curr_pos[:, 1], 1 - curr_pos[:, 1]))
            
            # 2. Find min circle distance for each point.
            # This is O(N^2)
            dists = np.linalg.norm(curr_pos[:, np.newaxis, :] - curr_pos[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_circle_dists = np.min(dists, axis=1) / 2.0
            
            # Global min
            r_min = np.min(np.minimum(wall_dists, min_circle_dists))
            
            # Update best
            if r_min > best_score:
                best_score = r_min
                best_positions = curr_pos.copy()
                
            # If r_min is very small or negative, we might be stuck.
            
            # Calculate forces
            forces = np.zeros_like(curr_pos)
            
            # Wall forces
            for i in range(N):
                x, y = curr_pos[i]
                if x < 0.5:
                    forces[i, 0] += 1.0
                else:
                    forces[i, 0] -= 1.0
                if y < 0.5:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
                    
            # Circle forces
            # Only push apart if they are the "bottleneck" (closest)
            # Or just push all pairs, weighted by proximity?
            # Simple: push all pairs apart.
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = curr_pos[i] - curr_pos[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-6:
                        d_vec = np.array([1e-6, 0])
                        d = 1e-6
                    # Force is stronger if d is small
                    # We want to increase d.
                    # Gradient of d w.r.t p_i is d_vec / d
                    # So we move p_i in direction d_vec / d
                    # And p_j in direction -d_vec / d
                    weight = 1.0 # Constant force? Or 1/d?
                    # Using 1/d helps resolve overlaps faster.
                    f = d_vec / (d * d) # 1/d
                    forces[i] += f
                    forces[j] -= f
                    
            # Apply forces with a learning rate
            # The learning rate should be small to avoid oscillation.
            # Since we are maximizing min distance, we can use a fixed small step.
            step_size = 0.01
            curr_pos += forces * step_size
            
            # Keep inside [0,1] (hard constraint)
            curr_pos = np.clip(curr_pos, 0, 1)
            
            # Optional: Random perturbation to escape local minima
            if step % 50 == 0:
                curr_pos += np.random.normal(0, 0.01, curr_pos.shape)
                curr_pos = np.clip(curr_pos, 0, 1)
                
    # Final check and return
    if best_positions is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Calculate final radii
    # We can set all radii to best_score.
    # Or we can try to adjust individual radii to gain more?
    # If we have slack in some directions, we can increase specific radii.
    # But uniform is a good approximation.
    
    radii = np.full(N, best_score)
    
    # Construct result
    result = np.column_stack((best_positions, radii))
    
    # Final validation
    # Check if any circle is outside or overlapping
    # If invalid, fallback
    for i in range(N):
        x, y, r = result[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
             return np.array([((i + 0.5) / 6, (j + 0.5) / 6, 0.5 / 6 - 1e-3) for i in range(6) for j in range(6)][:N])
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(result[i, :2] - result[j, :2])
            if dist < result[i, 2] + result[j, 2] - 1e-9:
                return np.array([((i + 0.5) / 6, (j + 0.5) / 6, 0.5 / 6 - 1e-3) for i in range(6) for j in range(6)][:N])
                
    return result
# EVOLVE-BLOCK-END
