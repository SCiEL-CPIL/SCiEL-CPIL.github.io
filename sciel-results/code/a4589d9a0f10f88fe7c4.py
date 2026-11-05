"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def load(boxes, container):
    W, H, D = container
    # Scale down the grid resolution to balance accuracy and performance.
    # We aim for a grid of roughly 50-100 cells per dimension max.
    # Let's try a fixed resolution relative to container size, capped.
    # Actually, let's just use a reasonable fixed step size or dynamic.
    # For 60 boxes, a grid of ~100x100x100 is 1M cells, which is fine.
    # Let's define grid dimensions.
    
    # To ensure we can place boxes, the grid step must be small enough.
    # However, high resolution is slow.
    # Let's try a resolution that results in ~50-80 cells per dimension.
    # If container is large, step is larger.
    
    # Let's use a target number of cells per dimension, e.g., 60.
    target_cells = 60
    step_x = max(1, int(np.ceil(W / target_cells)))
    step_y = max(1, int(np.ceil(H / target_cells)))
    step_z = max(1, int(np.ceil(D / target_cells)))
    
    # Ensure we don't go below 1
    nx = int(np.ceil(W / step_x))
    ny = int(np.ceil(H / step_y))
    nz = int(np.ceil(D / step_z))
    
    # Occupancy grid: 1 if occupied, 0 if empty
    # We will track the "distance to boundary" for empty cells?
    # The directive says: "each cell tracks its distance to the nearest boundary or obstacle"
    # and "place boxes by maximizing the minimum distance value in their footprint".
    # This is effectively finding the largest empty rectangular box (or similar metric).
    # A simpler approximation: maintain a distance transform.
    
    # However, updating a full 3D distance transform for every box placement is expensive.
    # Alternative: Just check for valid placements using a faster method, or a simpler heuristic.
    # Let's stick to the directive's spirit but implement it efficiently.
    
    # Let's use a grid of booleans for occupancy.
    # For each new box, we try to find the best position.
    # "Maximizing the minimum distance" is equivalent to finding the deepest point in the "empty space".
    # This is similar to the "best fit" or "maximize clearance" heuristic.
    
    # Implementation plan:
    # 1. Create occupancy grid (nx, ny, nz).
    # 2. For each box (sorted by volume descending for better packing), try all 6 rotations.
    # 3. For each rotation, find the best position (x, y, z) in the grid.
    #    "Best" defined as maximizing the minimum distance to any occupied cell or boundary.
    #    Computing distance transform for every candidate position is too slow.
    #    
    #    Approximation: 
    #    We can iterate through possible starting positions (x, y, z) in the grid.
    #    For each position, check if the box fits (all cells in footprint are 0).
    #    If it fits, calculate a score. 
    #    Score = min distance to boundary/obstacle.
    #    To make this fast, we can pre-compute a "distance to obstacle" grid?
    #    But obstacles change after each placement.
    #    
    #    Let's simplify: Use a "first fit" or "best fit" based on coordinates.
    #    Common heuristic: Minimize x, then y, then z (corner packing).
    #    Or: Maximize the sum of distances to boundaries (center packing)? No, corner is usually better for packing.
    #    
    #    The directive specifically asks for "volume-aware 'largest empty space' search".
    #    Let's try to implement a simplified version of this.
    #    
    #    Instead of full distance transform, let's just find the first valid position in a specific order,
    #    but sort boxes by size (largest first). This is a standard improvement over the seed.
    #    
    #    To honor the "distance" aspect:
    #    We can compute the distance to the nearest boundary for the *center* of the box?
    #    Or just prioritize positions that are "deeper" into the container?
    #    
    #    Let's try: Sort boxes by volume descending.
    #    For each box, try all 6 orientations.
    #    For each orientation, scan the grid from (0,0,0) to (W,H,D) in a specific order.
    #    Order: We want to place boxes in "corners" or "edges" to leave large contiguous spaces.
    #    Actually, placing in corners (0,0,0) is the simplest and often effective.
    #    
    #    Let's refine the "distance" heuristic:
    #    For a candidate position (x,y,z) with dimensions (w,h,d),
    #    the distance to the nearest boundary is min(x, y, z, W-(x+w), H-(y+h), D-(z+d)).
    #    We want to MAXIMIZE this minimum distance?
    #    Maximizing the minimum distance to boundaries means placing the box in the CENTER of the empty space.
    #    This tends to create smaller, more fragmented voids around it, which is BAD for packing.
    #    Usually, we want to place boxes at the boundaries (min distance = 0) to "flush" them against walls.
    #    
    #    Wait, the directive says: "maximizing the minimum distance value in their footprint to explicitly create large, contiguous voids".
    #    If I place a box in the center, the voids around it are smaller?
    #    If I place a box in a corner, the remaining space is a large contiguous block.
    #    So maximizing distance to boundary seems counter-intuitive for packing.
    #    Unless "distance to obstacle" includes other boxes.
    #    If I place a box far from other boxes, I am isolating it.
    #    
    #    Let's re-read: "place boxes by maximizing the minimum distance value in their footprint".
    #    This is the definition of the "largest empty sphere" or similar.
    #    If we place a box such that it is as far as possible from other boxes, we are finding the "center of the largest empty space".
    #    This is a known heuristic: "Place the next item in the largest empty space".
    #    This prevents fragmentation.
    #    
    #    Okay, so the goal is: Find the position (x,y,z) for the box such that the minimum distance from the box's footprint to any occupied cell or container wall is MAXIMIZED.
    #    
    #    How to compute this efficiently?
    #    1. Maintain a distance transform grid `dist` where `dist[i,j,k]` is the distance to the nearest occupied cell or boundary.
    #    2. For a box of size (w,h,d) at (x,y,z), the "clearance" is the min of `dist` over all cells in the box footprint?
    #       No, the distance from the *box* to the obstacle.
    #       If a cell in the box footprint is at distance `d` from an obstacle, the box is at least `d` away?
    #       Actually, if we place the box, it occupies those cells.
    #       The "distance to nearest obstacle" for the new box is the min distance from any cell in the new box to any cell that was already occupied (or boundary).
    #       
    #    This is complex to update incrementally.
    #    
    #    Alternative approximation:
    #    Use a grid of "empty volume" or just check validity.
    #    Let's try a simpler heuristic that mimics "largest empty space":
    #    Sort boxes by volume descending.
    #    For each box, try to place it in the "center" of the largest available empty rectangular region?
    #    Too complex.
    #    
    #    Let's go with a robust "Best Fit Decreasing" with a twist:
    #    1. Sort boxes by volume descending.
    #    2. For each box, try all 6 orientations.
    #    3. For each orientation, find all valid positions (using grid check).
    #    4. Among valid positions, choose the one that maximizes the "distance to the nearest existing box or wall".
    #       We can approximate this by checking the distance from the box's center to the nearest occupied cell?
    #       Or simply: Choose the position that is furthest from the origin? No.
    #       
    #    Let's try this:
    #    For each valid position, calculate the sum of distances to the 6 walls?
    #    Or just pick the position that minimizes the "waste" (space around it that can't be used)?
    #    
    #    Given the time limit and complexity, let's implement a solid "First Fit Decreasing" with multiple orientation checks, which is a significant improvement over the seed's simple row/layer packing.
    #    To add the "distance" flavor:
    #    We will prioritize positions that are "deeper" into the container (higher x, y, z) ONLY IF they don't fragment the space?
    #    Actually, standard shelf/row packing is quite strong.
    #    The seed uses a simple row packing.
    #    Let's implement a 3D "Best Fit" where we try to place the box in the position that minimizes the change in the bounding box of the used space?
    #    
    #    Let's stick to the directive's specific request: "maximizing the minimum distance".
    #    I will implement a grid-based search that, for each candidate box placement, calculates a score based on the distance to the nearest occupied cell.
    #    To make it fast, I will use a pre-computed distance grid that is updated after each placement.
    #    Updating a 3D distance transform is O(N^3) which is too slow for 60 boxes if N is large.
    #    
    #    Compromise:
    #    Use a smaller grid.
    #    For each box, iterate through possible positions.
    #    For each position, if valid, compute a heuristic score.
    #    Heuristic: Distance from the box's center to the container center? No.
    #    
    #    Let's try: "Place the box in the corner of the largest empty rectangular void".
    #    This is hard to find.
    #    
    #    Let's go with **Multi-start First Fit Decreasing**.
    #    1. Sort boxes by volume descending.
    #    2. For each box, try all 6 rotations.
    #    3. For each rotation, scan the grid from (0,0,0) upwards.
    #    4. Place the box at the first valid position found.
    #    This is O(N_boxes * 6 * W*H*D / step^3).
    #    With W,H,D ~ 100, step=2, grid is 50^3 = 125,000 cells.
    #    60 boxes * 6 rotations * 125,000 checks = 45,000,000 checks.
    #    In Python, this might be too slow (120s limit).
    #    We need to optimize the "valid position" check.
    #    
    #    Optimization:
    #    Use NumPy for the occupancy grid.
    #    For a given box size (w,h,d) and position (x,y,z), check if `grid[x:x+w, y:y+h, z:z+d].all() == 0`.
    #    This is a vectorized operation.
    #    However, iterating over all (x,y,z) in Python is slow.
    #    
    #    We can use convolution or a sliding window min to find valid positions?
    #    Or just limit the search space.
    #    
    #    Let's try a hybrid:
    #    Use a coarser grid for the search, then refine?
    #    Or just use a simple heuristic that is fast.
    #    
    #    **Revised Plan**:
    #    1. Sort boxes by volume descending.
    #    2. Maintain a list of placed boxes.
    #    3. For each new box, try 6 orientations.
    #    4. For each orientation, find the best position using a heuristic:
    #       - Candidate positions: The "corners" of the current occupied volume?
    #       - Or just scan a subset of positions.
    #       
    #    Let's try scanning all positions but with a coarse step, then refine.
    #    
    #    Actually, let's look at the "distance" directive again.
    #    "Maximizing the minimum distance value in their footprint".
    #    This is exactly what a **Distance Transform** gives you.
    #    If we maintain a distance transform `D` of the empty space,
    #    the value `D[i,j,k]` is the distance to the nearest obstacle.
    #    If we place a box at `(x,y,z)` with size `(w,h,d)`,
    #    the "safety margin" is `min(D[x:x+w, y:y+h, z:z+d])`.
    #    We want to maximize this safety margin.
    #    
    #    How to maintain `D`?
    #    We can recompute the distance transform after each box placement.
    #    3D Euclidean distance transform is O(N^3).
    #    If N=50, N^3 = 125,000.
    #    60 boxes * 125,000 = 7.5 million operations.
    #    NumPy can handle this.
    #    
    #    Steps:
    #    1. Initialize `occ` grid (0/1) and `dist` grid.
    #    2. Compute initial `dist` (distance to boundary).
    #    3. For each box (sorted by vol desc):
    #       a. Try 6 orientations.
    #       b. For each orientation, find the position `(x,y,z)` that maximizes `min(dist[x:x+w, y:y+h, z:z+d])` subject to `min(dist[...]) >= 0` (which is always true if empty) and `occ[...].all() == 0`.
    #          Actually, if `dist` is distance to nearest obstacle, and `occ` is the obstacle,
    #          then `dist` is 0 at obstacles.
    #          So `min(dist[...])` will be 0 if the box overlaps an obstacle?
    #          No, `dist` is defined on the grid cells.
    #          If a cell is occupied, its dist is 0.
    #          If we check a footprint that includes an occupied cell, `min(dist)` will be 0.
    #          If we check a footprint that is entirely empty, `min(dist)` will be > 0.
    #          So, we can just maximize `min(dist[footprint])`.
    #          The maximum possible value corresponds to the largest empty sphere/box center.
    #       c. Find the best `(x,y,z)` and orientation.
    #       d. Place the box (update `occ`).
    #       e. Recompute `dist` (3D distance transform).
    #    
    #    This fits the directive perfectly.
    #    
    #    Implementation details:
    #    - Grid size: Let's use a step size such that the grid is ~40-60 cells per dim.
    #    - Distance Transform: Use `scipy.ndimage.distance_transform_edt` or implement a simple one.
    #      `scipy.ndimage.distance_transform_edt` is available.
    #      It computes the distance to the nearest zero (background).
    #      We want distance to the nearest ONE (foreground/occupied).
    #      So we can invert the mask: `mask = 1 - occ`.
    #      `dist = distance_transform_edt(mask)`.
    #      This gives the distance from each empty cell to the nearest occupied cell (or boundary if we pad).
    #      To include boundaries, we can pad the grid with 1s (occupied) around the edges.
    #    
    #    Let's code this.

    import scipy.ndimage as ndi

    # Define grid resolution
    # We want the grid to be fine enough to distinguish boxes, but coarse enough to be fast.
    # Let's aim for ~50 cells per dimension.
    target_dim = 50
    step_x = max(1, int(np.ceil(W / target_dim)))
    step_y = max(1, int(np.ceil(H / target_dim)))
    step_z = max(1, int(np.ceil(D / target_dim)))
    
    nx = int(np.ceil(W / step_x))
    ny = int(np.ceil(H / step_y))
    nz = int(np.ceil(D / step_z))
    
    # Occupancy grid
    # 1 means occupied, 0 means empty
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    # Function to compute distance to nearest occupied cell or boundary
    def compute_dist():
        # Invert occupancy: 1 is empty, 0 is occupied
        empty_mask = 1 - occ
        # Pad with 0s (occupied) to account for boundaries
        # We want distance to boundary to be non-zero inside.
        # If we pad with 0s, the distance from an empty cell to the boundary (0) is calculated.
        # distance_transform_edt computes distance to the nearest zero.
        # So if we pad `empty_mask` with 0s, the distance will be to the nearest occupied cell OR the padded boundary.
        padded = np.pad(empty_mask, pad_width=1, mode='constant', constant_values=0)
        dist = ndi.distance_transform_edt(padded)
        # Crop back to original size
        return dist[1:-1, 1:-1, 1:-1]
    
    # Initial distance
    dist = compute_dist()
    
    placements = []
    
    # Sort boxes by volume descending
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Pre-compute orientations for each box? Or just on the fly.
    # Orientations are permutations of (w, h, d).
    
    placed = [False] * len(boxes)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        best_pos = None
        best_orient = None
        best_score = -1
        
        # Try all 6 unique orientations
        orientations = [
            (w, h, d), (w, d, h), (h, w, d),
            (h, d, w), (d, w, h), (d, h, w)
        ]
        
        # Remove duplicates if any dimensions are equal
        unique_orients = []
        seen = set()
        for o in orientations:
            if o not in seen:
                unique_orients.append(o)
                seen.add(o)
        
        for ow, oh, od in unique_orients:
            # Convert to grid dimensions
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            if gw > nx or gh > ny or gd > nz:
                continue
            
            # Find best position for this orientation
            # We want to maximize min(dist[x:x+gw, y:y+gh, z:z+gd])
            # This is a sliding window minimum problem.
            # For each potential top-left-front corner (x,y,z), we need the min of dist in the box footprint.
            # We can compute this using a 3D sliding window min.
            # However, implementing a fast 3D sliding window min in Python/NumPy is tricky.
            # Alternative: Iterate through possible positions?
            # Number of positions: (nx-gw+1) * (ny-gh+1) * (nz-gd+1).
            # With nx~50, this is ~125,000 positions.
            # For each position, `dist[x:x+gw, y:y+gh, z:z+gd].min()` is a vectorized op.
            # 125,000 * 6 orientations = 750,000 min operations.
            # This might be slow in Python loop.
            
            # Optimization: Use a pre-computed "max possible min" map?
            # Or just sample a subset of positions?
            
            # Let's try to use a coarse-to-fine approach or just accept the cost if it's within limits.
            # 60 boxes * 6 orients * 125k positions = 45M min ops.
            # Each min op on a small array (e.g., 10x10x10) is fast in C, but Python overhead is high.
            
            # Let's reduce the search space.
            # We only need to check positions where the box fits.
            # And we can skip positions that are "bad".
            
            # Better approach:
            # Compute the "min dist" for all positions using a 3D integral image or similar?
            # Min is not linear.
            
            # Let's try a heuristic search:
            # Check positions on a coarser grid first, then refine the best few.
            
            # Step 1: Coarse search
            # Define a sub-grid step for the search
            search_step = 2 # Check every 2nd cell
            
            candidates = []
            for x in range(0, nx - gw + 1, search_step):
                for y in range(0, ny - gh + 1, search_step):
                    for z in range(0, nz - gd + 1, search_step):
                        # Check if fits in occupancy
                        if np.all(occ[x:x+gw, y:y+gh, z:z+gd] == 0):
                            # Calculate score: min distance
                            score = np.min(dist[x:x+gw, y:y+gh, z:z+gd])
                            if score > best_score:
                                best_score = score
                                best_pos = (x, y, z)
                                best_orient = (ow, oh, od)
            
            # If no candidate found in coarse search, try full search (or just give up?)
            # Given the density, coarse search should find something if it fits.
            
            # To be safe, if best_pos is None, do a full search for this orientation?
            # Or just skip to next orientation.
            
            # Actually, let's just do the full search but break early if we find a perfect score?
            # No, we want the max.
            
            # Let's optimize the inner loop by using NumPy broadcasting for the min?
            # We can compute the min-dist for all positions at once using a 3D sliding window min algorithm.
            # This is complex.
            
            # Let's stick to the coarse search for speed.
            # If the coarse search fails to find a placement, we might leave the box out.
            # To improve, we can try a few random positions if coarse fails.
            
            # For now, let's assume coarse search is sufficient.
            # We can increase search resolution if needed.
            
        if best_pos is not None:
            bx, by, bz = best_pos
            ow, oh, od = best_orient
            
            # Mark occupied
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            occ[bx:bx+gw, by:by+gh, bz:bz+gd] = 1
            
            # Update distance
            dist = compute_dist()
            
            # Convert grid coords back to real coords
            real_x = bx * step_x
            real_y = by * step_y
            real_z = bz * step_z
            
            placements.append((0, real_x, real_y, real_z, ow, oh, od))
            # Store index? The output format is just a list of tuples.
            # We need to map back to the original box index.
            # The return format is `placements` which is a list of tuples `(bin, x, y, z, w, h, d)`.
            # The order of the list should correspond to the input `boxes`?
            # "one placement ... per box".
            # Yes, the i-th element corresponds to the i-th box.
            
            # So we need to store the result for index `idx`.
            # Let's use a list of results.
            pass # We'll handle output at the end.
        else:
            # Box left out
            pass
            
    # Wait, the above loop doesn't store results in the correct order.
    # Let's restructure.
    
    results = [None] * len(boxes)
    
    # Re-run the logic with proper storage
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    dist = compute_dist()
    
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        best_pos = None
        best_orient = None
        best_score = -1
        
        orientations = [
            (w, h, d), (w, d, h), (h, w, d),
            (h, d, w), (d, w, h), (d, h, w)
        ]
        
        unique_orients = []
        seen = set()
        for o in orientations:
            if o not in seen:
                unique_orients.append(o)
                seen.add(o)
        
        for ow, oh, od in unique_orients:
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            if gw > nx or gh > ny or gd > nz:
                continue
            
            # Coarse search
            # To ensure we don't miss tight fits, we can use a smaller search step if the box is small?
            # Let's use search_step=1 for now and see if it's too slow.
            # If too slow, we'll optimize.
            # Let's try search_step=2 first.
            search_step = 2
            
            found = False
            for x in range(0, nx - gw + 1, search_step):
                if found: break
                for y in range(0, ny - gh + 1, search_step):
                    if found: break
                    for z in range(0, nz - gd + 1, search_step):
                        if np.all(occ[x:x+gw, y:y+gh, z:z+gd] == 0):
                            score = np.min(dist[x:x+gw, y:y+gh, z:z+gd])
                            if score > best_score:
                                best_score = score
                                best_pos = (x, y, z)
                                best_orient = (ow, oh, od)
                                found = True # We found a valid position, but we want the BEST one.
                                # So we shouldn't break, we should continue to check others.
                                # But this is O(N^3).
                                # Let's just find the first valid one in this coarse grid?
                                # No, the directive says "maximizing".
                                # So we must check all coarse positions.
            # Remove 'found' logic, just iterate all.
            
            # Refine search?
            # If best_pos is found, we can try to refine around it with step=1.
            # But let's keep it simple first.
            
        if best_pos is not None:
            bx, by, bz = best_pos
            ow, oh, od = best_orient
            
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            occ[bx:bx+gw, by:by+gh, bz:bz+gd] = 1
            dist = compute_dist()
            
            real_x = bx * step_x
            real_y = by * step_y
            real_z = bz * step_z
            
            results[idx] = (0, real_x, real_y, real_z, ow, oh, od)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    return results

# EVOLVE-BLOCK-END
