"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # Initial guess: Grid placement
    # For N=26, a 5x5 grid is 25 points. We can use a 5x5 grid + 1 center point, 
    # or a 6x5 grid taking 26 points. 
    # Let's use a 5x5 grid for the first 25 and place the 26th in a gap or just use a 6x5 grid.
    # A 5x5 grid of 25 circles leaves 1 spot. 
    # Better: Use a random jittered grid to avoid symmetry traps.
    
    # Create a 5x5 grid for 25 circles
    grid_side = 5
    x_coords = np.linspace(1.0/(2*grid_side), 1.0 - 1.0/(2*grid_side), grid_side)
    y_coords = np.linspace(1.0/(2*grid_side), 1.0 - 1.0/(2*grid_side), grid_side)
    
    init_points = []
    for i in range(grid_side):
        for j in range(grid_side):
            init_points.append((x_coords[i], y_coords[j]))
            
    # We have 25 points. Add one more in the center of the square? 
    # Or just use a 6x5 grid and take 26.
    # Let's stick to 5x5 + 1 at (0.5, 0.5) might cause overlap with center.
    # Let's use a 6x5 grid.
    grid_side_x = 6
    grid_side_y = 5
    x_coords = np.linspace(1.0/(2*grid_side_x), 1.0 - 1.0/(2*grid_side_x), grid_side_x)
    y_coords = np.linspace(1.0/(2*grid_side_y), 1.0 - 1.0/(2*grid_side_y), grid_side_y)
    
    init_points = []
    for i in range(grid_side_x):
        for j in range(grid_side_y):
            if len(init_points) < N:
                init_points.append((x_coords[i], y_coords[j]))
                
    # Ensure we have exactly N points
    while len(init_points) < N:
        # Add random points if grid wasn't enough (should not happen for 6x5=30 >= 26)
        init_points.append((0.5, 0.5))
        
    init_points = init_points[:N]
    
    # State: centers (N, 2) and radii (N,)
    centers = np.array([p[0] for p in init_points]).reshape(-1, 1)
    centers = np.hstack([centers, np.array([p[1] for p in init_points]).reshape(-1, 1)])
    radii = np.full(N, 0.05)
    
    # PBD Solver
    # Parameters
    growth_rate = 0.001
    projection_iterations = 50
    total_iterations = 5000
    
    for it in range(total_iterations):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # 1. Grow radii
        # We want to maximize sum of radii. 
        # In PBD, we can try to increase radii uniformly.
        # However, simple uniform growth might not be optimal if some circles are stuck.
        # A better heuristic: grow all, then project.
        radii += growth_rate
        
        # 2. Project constraints
        for _ in range(projection_iterations):
            # Boundary Projection
            # For each circle, ensure it is inside [0,1]^2
            # If x < r, set x = r. If x > 1-r, set x = 1-r.
            # This is a hard projection.
            
            # Overlap Projection
            # For each pair (i, j), if dist < r_i + r_j, push them apart.
            
            # We can do these in parallel or sequential. Sequential is easier to code.
            # To speed up, we can vectorize the overlap check.
            
            # --- Boundary ---
            # x_min = radii
            # x_max = 1 - radii
            # y_min = radii
            # y_max = 1 - radii
            
            # If center is out, clamp it.
            centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
            
            # --- Overlaps ---
            # Vectorized distance calculation
            # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            # dist = np.linalg.norm(diff, axis=2)
            # This creates an NxN matrix. For N=26, this is small (26x26=676).
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            
            # required_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            req_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Find overlaps: dist < req_dist
            # We only care about i < j to avoid double counting and self-interaction.
            # Create a mask for i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Get indices of overlapping pairs
            overlap_indices = np.argwhere(mask & (dist < req_dist))
            
            if len(overlap_indices) > 0:
                for i, j in overlap_indices:
                    # Calculate push vector
                    d = dist[i, j]
                    if d < 1e-10:
                        # If centers are same, push randomly
                        dx = np.random.uniform(-0.1, 0.1)
                        dy = np.random.uniform(-0.1, 0.1)
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm == 0:
                            continue
                        dx /= norm
                        dy /= norm
                    else:
                        dx = (centers[j, 0] - centers[i, 0]) / d
                        dy = (centers[j, 1] - centers[i, 1]) / d
                    
                    # Overlap amount
                    overlap = req_dist[i, j] - d
                    
                    # Push each circle by half the overlap
                    # This is standard PBD for equal mass.
                    # For unequal mass (radius), we could weight by radius, but equal mass is simpler and stable.
                    push = overlap / 2.0
                    
                    centers[i, 0] -= dx * push
                    centers[i, 1] -= dy * push
                    centers[j, 0] += dx * push
                    centers[j, 1] += dy * push
                    
                    # After pushing, we might have violated boundaries again.
                    # The next iteration of the projection loop will fix boundaries.
                    
        # 3. Clamp radii to ensure they are valid given the current centers?
        # Actually, the boundary projection handles this by moving centers.
        # But if a circle is squeezed into a corner, its max radius is limited by min(x, 1-x, y, 1-y).
        # If we grew r beyond what fits in the current position, the boundary projection
        # will push the center away from the wall, potentially into another circle.
        # This iterative process settles into a jammed state.
        
        # Optional: If a circle is extremely small or stuck, we might want to handle it.
        # But for now, this should work.
        
    # Final cleanup: Ensure strict validity
    # Sometimes numerical errors leave tiny overlaps.
    for _ in range(100):
        changed = False
        # Boundary check
        for i in range(N):
            min_dist = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > min_dist:
                radii[i] = min_dist
                changed = True
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                req = radii[i] + radii[j]
                if dist < req:
                    # Shrink the larger one? Or both?
                    # To be safe and ensure validity, shrink the larger one.
                    if radii[i] >= radii[j]:
                        radii[i] = (dist - radii[j]) * 0.9999
                        if radii[i] < 0: radii[i] = 0
                    else:
                        radii[j] = (dist - radii[i]) * 0.9999
                        if radii[j] < 0: radii[j] = 0
                    changed = True
                    
        if not changed:
            break
            
    # Construct output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i, 0]
        result[i, 1] = centers[i, 1]
        result[i, 2] = radii[i]
        
    return result

# EVOLVE-BLOCK-END

