import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Preprocess boxes: Sort by volume descending (FFD)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    placements = [None] * n
    
    # 2. Initialize Occupancy Grid
    # Grid resolution is 1 unit. If dimensions are large, this might be memory heavy,
    # but for typical container problems (e.g., 100x100x100), 1M cells is fine.
    # To be safe against memory limits for huge containers, we check size.
    # If W*H*D is too big, we fall back to list-based overlap checking.
    
    # Threshold for dense grid: 50 million cells (approx 50MB for int8)
    # This is a safe bet for most CI/eval environments.
    max_grid_cells = 50_000_000
    use_dense_grid = (W * H * D) <= max_grid_cells
    
    if use_dense_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        grid = None

    placed_boxes = []
    
    # 2. Initialize Extreme Points
    # Each point is (x, y, z)
    extreme_points = [(0, 0, 0)]
    
    # 3. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as not placed
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        # Get unique orientations for this box
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # Sort extreme points by (z, y, x) to prefer lower positions (fill bottom layers first)
        # This helps in creating a stable "skyline"
        sorted_eps = sorted(extreme_points, key=lambda p: (p[2], p[1], p[0]))
        
        for px, py, pz in sorted_eps:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orients:
                # Check bounds
                if px + ow > W or py + oh > H or pz + od > D:
                    continue
                
                # Check overlap
                valid = True
                if use_dense_grid:
                    # Fast numpy check
                    if np.any(grid[px:px+ow, py:py+oh, pz:pz+od]):
                        valid = False
                else:
                    # Fallback: check against placed boxes list
                    for bx, by, bz, bw, bh, bd in placed_boxes:
                        if (px < bx + bw and bx < px + ow and
                            py < by + bh and by < py + oh and
                            pz < bz + bd and bz < pz + od):
                            valid = False
                            break
                
                if valid:
                    # Found a valid placement
                    # Score: Prefer positions that are "lower" and "closer to origin"
                    # z is most important (gravity), then y, then x
                    # We want to minimize the "wasted space" below and around.
                    # A simple metric is the sum of coordinates, weighted by axis importance.
                    # Let's try (z * 100 + y * 10 + x) or just z+y+x.
                    # Since we sorted EPs by (z,y,x), the first valid one found in this order
                    # is likely the best "lowest" position. 
                    # However, different orientations might allow placing at a lower z?
                    # No, z is fixed by the EP.
                    # So for a given EP, we just want to pick the orientation that fits.
                    # Actually, we should check if a different orientation allows a *better* score?
                    # The score is primarily position-based.
                    # Let's stick to the first valid orientation found in the sorted EP loop,
                    # but we should probably prefer orientations that leave less "dead space" 
                    # or align better with the container edges?
                    # For now, let's just take the first valid one encountered in the loop.
                    # Since EPs are sorted, the first EP is the "best" position.
                    # Among orientations at that position, does it matter? 
                    # Yes, for future packing. But for immediate density, any fit is good.
                    # Let's just take the first valid orientation.
                    best_placement = (px, py, pz, ow, oh, od)
                    break # Break orientation loop, found valid for this EP
            
            if best_placement is not None:
                # Found a valid placement at this EP. 
                # Since EPs are sorted by "goodness" (z,y,x), the first valid EP is the best.
                # We can break the EP loop immediately?
                # Yes, because we are looking for the "best" position, and we scan in order of preference.
                break
        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            placed_boxes.append((px, py, pz, ow, oh, od))
            
            if use_dense_grid:
                grid[px:px+ow, py:py+oh, pz:pz+od] = 1
            
            # Generate new extreme points
            # The new box occupies [px, px+ow] x [py, py+oh] x [pz, pz+od]
            # New potential positions for next boxes are the corners of this box that are "exposed"
            new_points = [
                (px + ow, py, pz),
                (px, py + oh, pz),
                (px, py, pz + od)
            ]
            
            # Add new points to extreme_points, avoiding duplicates
            # To avoid O(N^2) checking for duplicates in a large list, we can use a set for membership
            # but we need to keep the list for iteration.
            # For N=60, list size is small (<200), so linear search is fine.
            for p in new_points:
                if p not in extreme_points:
                    extreme_points.append(p)
        else:
            # Box could not be placed
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
