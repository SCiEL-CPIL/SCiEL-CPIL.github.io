"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin for safety
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute orientations for each box
    all_orientations = []
    for w, h, d in boxes:
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        ors = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                ors.append(perm)
        all_orientations.append(ors)

    # Grid resolution: aim for ~10x10x10 grid for coarse structure
    # But ensure at least 1 cell per unit if container is small?
    # Let's use a fixed resolution that balances memory and speed.
    # 60 boxes, so grid shouldn't be too huge.
    # Let's try 15x15x15 or similar.
    grid_res = 15
    grid_w = int(W / grid_res) + 1
    grid_h = int(H / grid_res) + 1
    grid_d = int(D / grid_res) + 1
    
    # Cap grid size to avoid memory issues if container is huge
    max_cells = 5000
    if grid_w * grid_h * grid_d > max_cells:
        scale = (max_cells / (grid_w * grid_h * grid_d)) ** (1/3)
        grid_w = max(1, int(grid_w * scale))
        grid_h = max(1, int(grid_h * scale))
        grid_d = max(1, int(grid_d * scale))
        
    # Ensure grid covers the whole container
    cell_w = W / grid_w
    cell_h = H / grid_h
    cell_d = D / grid_d
    
    # Occupancy grid: 1 if occupied, 0 if empty
    # We will use a 3D numpy array
    occ_grid = np.zeros((grid_w, grid_h, grid_d), dtype=np.uint8)
    
    # List of placed boxes for exact overlap check
    # Format: (x, y, z, w, h, d, idx)
    placed = []
    results = [None] * len(boxes)
    
    # Strategy: Best Fit Decreasing with Voxel-Guided Candidate Selection
    # 1. Sort boxes by volume descending
    box_order = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def update_grid(x, y, z, w, h, d):
        # Mark cells occupied by this box
        x0 = int(x / cell_w)
        y0 = int(y / cell_h)
        z0 = int(z / cell_d)
        x1 = int((x + w) / cell_w)
        y1 = int((y + h) / cell_h)
        z1 = int((z + d) / cell_d)
        
        # Clamp to grid bounds
        x0 = max(0, min(x0, grid_w - 1))
        y0 = max(0, min(y0, grid_h - 1))
        z0 = max(0, min(z0, grid_d - 1))
        x1 = max(0, min(x1, grid_w - 1))
        y1 = max(0, min(y1, grid_h - 1))
        z1 = max(0, min(z1, grid_d - 1))
        
        # Mark the volume. Note: if x1 == x0, we still mark that cell.
        # We want to mark all cells that intersect the box.
        # If box starts at 0 and ends at cell_w, it occupies cell 0.
        # int(0/cell_w)=0, int(cell_w/cell_w)=1. So range(0, 1) is correct.
        occ_grid[x0:x1+1, y0:y1+1, z0:z1+1] = 1

    def get_candidate_positions():
        """
        Identify promising candidate positions based on the occupancy grid.
        We look for the 'corners' of the largest empty spaces.
        A simple heuristic: find the first empty cell in a space-filling curve or just scan.
        Better: Find the largest empty rectangular block in the grid and use its corners.
        For speed, we'll just collect the positions of the first few empty cells in a specific order
        and the corners of the bounding box of large empty regions.
        
        To be more strategic:
        1. Find all empty cells.
        2. Cluster them? Too slow.
        3. Just use the extremal points of the occupied set + origin.
           This is what the parent does. 
           
        Let's try: Use the grid to filter candidates.
        Generate candidates from extremal points of placed boxes (as before).
        Then, for each candidate, check if the grid indicates a large empty space nearby.
        Actually, let's just use the grid to *generate* candidates.
        
        Strategy: 
        - Find the largest empty axis-aligned cuboid in the grid (approximate).
        - Use its 8 corners as primary candidates.
        - Also keep the origin and some extremal points.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Extremal points from placed boxes
        for px, py, pz, pw, ph, pd, _ in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Also add the "back" corners?
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            
        # Filter candidates: Keep only those that are inside the container
        valid_candidates = []
        for cx, cy, cz in candidates:
            if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                valid_candidates.append((cx, cy, cz))
                
        # Sort candidates by "potential": 
        # We want to prioritize positions that are in "large" empty spaces.
        # Approximate by checking the grid value at the candidate point?
        # If the cell at (cx, cy, cz) is empty, it's good.
        # But we want to find the *largest* empty space.
        
        # Let's do a simple "Largest Empty Box" approximation on the grid.
        # This is O(N^3) on the grid, which is 15^3 = 3375. Very fast.
        # We find the largest empty sub-cuboid in the grid.
        
        # Function to find largest empty box in a binary grid
        def largest_empty_box(grid):
            # grid is 3D numpy array, 1=occupied, 0=empty
            # We want max volume sub-cuboid with all 0s.
            # This is hard in 3D. 
            # Approximation: Project to 2D? Or just scan?
            # Let's just scan for the first large empty region.
            # Or simply: find the centroid of the empty space?
            
            # For now, let's just return the corners of the bounding box of all empty cells.
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) == 0:
                return None
            
            min_x, min_y, min_z = empty_cells.min(axis=0)
            max_x, max_y, max_z = empty_cells.max(axis=0)
            
            # Convert back to world coordinates
            x0 = min_x * cell_w
            y0 = min_y * cell_h
            z0 = min_z * cell_d
            x1 = (max_x + 1) * cell_w
            y1 = (max_y + 1) * cell_h
            z1 = (max_z + 1) * cell_d
            
            # Clamp to container
            x0 = max(0, min(x0, W))
            y0 = max(0, min(y0, H))
            z0 = max(0, min(z0, D))
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            z1 = max(0, min(z1, D))
            
            # Return the 8 corners of this bounding box
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x0, y1, z0),
                (x1, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x0, y1, z1),
                (x1, y1, z1)
            ]
            return corners
        
        grid_corners = largest_empty_box(occ_grid)
        if grid_corners:
            for c in grid_corners:
                valid_candidates.append(c)
                
        # Deduplicate and limit
        unique_candidates = list(set(valid_candidates))
        
        # If too many, prune
        if len(unique_candidates) > 100:
            # Keep the ones closest to the origin? Or just take the first 100?
            # Let's sort by distance to origin to prioritize compactness
            unique_candidates.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
            unique_candidates = unique_candidates[:100]
            
        return unique_candidates

    placed_vol = 0
    
    # Run multiple attempts with different sorting strategies
    strategies = [
        ("vol", lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]),
        ("max", lambda i: max(boxes[i])),
        ("min", lambda i: min(boxes[i]))
    ]
    
    best_score = -1
    best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    for strat_name, sort_key in strategies:
        if time.time() > deadline - 5:
            break
            
        # Reset state
        occ_grid[:] = 0
        placed = []
        results = [None] * len(boxes)
        placed_vol = 0
        
        box_order = sorted(range(len(boxes)), key=sort_key, reverse=True)
        
        for idx in box_order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            ors = all_orientations[idx]
            
            candidates = get_candidate_positions()
            
            best_score_val = float('inf')
            best_placement = None
            
            for cx, cy, cz in candidates:
                for ow, oh, od in ors:
                    # Bounds check
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Overlap check
                    overlap = False
                    for px, py, pz, pw, ph, pd, _ in placed:
                        if not (cx + ow <= px or px + pw <= cx or 
                                cy + oh <= py or py + ph <= cy or 
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if overlap:
                        continue
                    
                    # Score: Minimize distance to origin (compactness)
                    # But also prefer positions that are in "larger" empty spaces?
                    # The candidate generation already biases towards large empty spaces.
                    # So just use compactness.
                    score = (cx + ow / 2.0)**2 + (cy + oh / 2.0)**2 + (cz + od / 2.0)**2
                    
                    if score < best_score_val:
                        best_score_val = score
                        best_placement = (cx, cy, cz, ow, oh, od)
                        
            if best_placement:
                cx, cy, cz, ow, oh, od = best_placement
                placed.append((cx, cy, cz, ow, oh, od, idx))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
                placed_vol += ow * oh * od
                update_grid(cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, w, h, d)
                
        # Finalize results
        for i in range(len(boxes)):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)
                
        if placed_vol > best_score:
            best_score = placed_vol
            best_results = results
            
    return best_results
# EVOLVE-BLOCK-END
