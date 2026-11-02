"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Geometric Initialization (Hexagonal Grid) + Local Greedy Search
    
    # 1. Generate a hexagonal lattice of 26 points
    # Hexagonal packing is denser than square packing.
    # We want to fit 26 points. A 5x5 square is 25. 
    # Let's generate a hex grid and pick the 26 best positioned or just generate a 5x5 and add one?
    # Better: Generate a hex grid with enough points, then select 26 that are well-distributed.
    
    # Generate hex points
    points = []
    rows = 6
    cols = 6
    # Hex spacing
    # Let's just create a grid and optimize from there. 
    # A 5x5 grid has 25. We need 1 more. 
    # Let's use a 5x5 grid, but shift it to maximize space, then add a 26th circle in the largest gap?
    # Or better: Start with a 5x5 grid of 25 circles, and split the largest one? No, that reduces sum of radii.
    
    # Let's try a specific known good configuration for 26 circles.
    # Often, a 5x5 grid with the center removed and 2 added? No.
    # Let's stick to a robust generic method: 
    # Initialize with a hexagonal lattice scaled to fit.
    
    # Hex lattice generation
    # Side length of hexagon cell approx 1/sqrt(3) for unit area density?
    # Let's just create a grid of candidates and pick 26.
    
    # Candidate grid: 8x8 = 64 points
    candidates = []
    for i in range(8):
        for j in range(8):
            x = (i + 0.5) / 8.0
            y = (j + 0.5) / 8.0
            # Shift odd rows to create hex pattern
            if j % 2 == 1:
                x += 0.5 / 8.0
            candidates.append([x, y])
            
    candidates = np.array(candidates)
    
    # We need to select 26 points from these 64 that maximize the minimum distance between them (or sum of radii).
    # This is a farthest-point sampling problem.
    
    selected = []
    # Start with a corner to ensure coverage
    selected.append(candidates[0]) # (0.0625, 0.0625)
    
    # Farthest point sampling
    for _ in range(N - 1):
        best_dist = -1
        best_idx = -1
        for k, c in enumerate(candidates):
            if c in selected:
                continue
            # Min distance to any selected point
            d_min = min(np.linalg.norm(c - s) for s in selected)
            if d_min > best_dist:
                best_dist = d_min
                best_idx = k
        selected.append(candidates[best_idx])
        
    centers = np.array(selected)
    
    # Initial radii
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        # Max radius is limited by distance to others / 2 and boundaries
        # Note: This assumes others have radius 0 initially for the first pass, 
        # but we can do a simultaneous update or just clamp.
        # For a better init, let's assume equal radii first?
        # No, let's compute max possible radius for each circle assuming others are points, 
        # then scale down to ensure no overlap.
        
        # Simplest robust init: 
        # r_i = min( min_j (d_ij - r_j) , boundary_dist )
        # This is circular. 
        # Let's just set r_i = 0.1 * min_dist_to_neighbor / 2 and refine.
        
        d_min = np.min(dists[mask])
        bound_r = min(centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
        radii[i] = min(d_min / 2.0, bound_r)
        
    # 2. Local Greedy Search
    # Move each circle slightly to increase the sum of radii.
    # Objective: Maximize sum(r_i)
    # r_i = min( min_j (d_ij - r_j), boundary_dist_i )
    
    # We iterate: 
    # For each circle i:
    #   Try moving it in 8 directions + stay.
    #   Calculate new r_i given new position (keeping other r_j fixed).
    #   If sum(r) increases, accept.
    
    # To make it faster and more effective, we can also adjust radii directly?
    # No, position is the main variable.
    
    directions = [
        [0, 0], [0.01, 0], [-0.01, 0], [0, 0.01], [0, -0.01],
        [0.007, 0.007], [-0.007, 0.007], [0.007, -0.007], [-0.007, -0.007]
    ]
    
    max_iter = 500
    for it in range(max_iter):
        improved = False
        for i in range(N):
            current_r_sum = np.sum(radii)
            best_move = None
            best_r_sum = current_r_sum
            
            for dx, dy in directions:
                new_c = centers[i] + np.array([dx, dy])
                # Clip to bounds (radius will handle it, but keep center inside for sanity)
                new_c = np.clip(new_c, 0.0, 1.0)
                
                # Calculate new radius for i
                dists = np.linalg.norm(centers - new_c, axis=1)
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                
                # Constraint: d_ij >= r_i + r_j => r_i <= d_ij - r_j
                # We need r_i > 0. If d_ij <= r_j, r_i can be 0.
                r_constraint = np.min(dists[mask] - radii[mask])
                bound_constraint = min(new_c[0], 1.0 - new_c[0], new_c[1], 1.0 - new_c[1])
                
                new_r_i = max(0.0, min(r_constraint, bound_constraint))
                
                # New sum
                new_r_sum = current_r_sum - radii[i] + new_r_i
                
                if new_r_sum > best_r_sum:
                    best_r_sum = new_r_sum
                    best_move = (new_c, new_r_i)
                    
            if best_move is not None:
                centers[i], radii[i] = best_move
                improved = True
                
        if not improved:
            break
            
    return np.column_stack((centers, radii))
# EVOLVE-BLOCK-END
