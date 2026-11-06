"""Improved 3D container loading with rotation-aware shelf packing."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    # Sort boxes by volume descending, then by max dimension descending
    box_list = list(enumerate(boxes))
    box_list.sort(key=lambda t: (-(t[1][0]*t[1][1]*t[1][2]), -max(t[1])))
    
    # We'll use a shelf-based approach with rotation.
    # For each box, try all 6 orientations and place it in the first shelf where it fits.
    # Shelves are horizontal layers (z-direction). Within a shelf, we pack along x, advancing y when x overflows.
    
    # State: list of shelves. Each shelf has:
    #   z_start, height (max h in shelf), y_pos, row_height, x_pos, row_width_limit? 
    # Actually, let's use a simpler "next-fit decreasing" with rotation.
    
    # Alternative: Use a priority queue of free spaces.
    # Let's implement a simple "First Fit Decreasing" with rotation on a 3D grid of free rectangles.
    
    # Simpler approach: shelf packing with rotation.
    # Shelves are defined by z-levels. Each shelf has a height.
    # Within a shelf, we place boxes side by side in x. When x overflows, we start a new row (increase y).
    # When y overflows, we start a new shelf (increase z).
    
    # But this is similar to the seed. Let's add rotation and better ordering.
    
    # Let's try: for each box (sorted by volume desc), try all 6 orientations.
    # For each orientation, try to place it in the first available position using a shelf heuristic.
    # We maintain a list of shelves. Each shelf: (z_start, shelf_height, y_cursor, row_max_h, x_cursor)
    # When placing a box with dims (w, h, d) at position (x, y, z):
    #   It occupies [x, x+w] x [y, y+h] x [z, z+d]
    #   In shelf packing, z is the shelf's z_start, and d must be <= remaining depth? No, d is the depth into the container.
    
    # Actually, let's redefine: shelves are along the z-axis. Each shelf has a fixed z_start and height (max h of boxes in it).
    # Within a shelf, we pack in the x-y plane. We use rows: each row has a fixed y_start and height (max h in row).
    # Wait, h is the vertical dimension within the shelf? Let's be careful.
    
    # Let's use: z is the depth (front-to-back), x is width, y is height.
    # Shelf packing: shelves are stacked in z. Each shelf has a z_start and a depth (d).
    # Within a shelf, we pack in the x-y plane. Rows are stacked in y. Each row has a y_start and height (h).
    # Within a row, boxes are placed side by side in x.
    
    # So for a box with orientation (w, h, d):
    #   d is the shelf depth contribution
    #   h is the row height contribution
    #   w is the x-extent
    
    # State:
    #   shelves: list of (z_start, shelf_depth_used, rows)
    #   Actually, let's simplify: 
    #   current_shelf_z, current_shelf_depth, current_row_y, current_row_height, current_x
    
    # This is getting complex. Let's use a different approach: 
    # "Best Fit" with a list of free spaces.
    
    # Initialize with one free space: the entire container.
    # For each box, try all 6 orientations.
    # For each orientation, find the free space where the box fits with the least waste.
    # If placed, split the free space into up to 3 new free spaces.
    
    # This is a standard 3D bin packing heuristic.
    
    free_spaces = [(0, 0, 0, W, H, D)]  # (x, y, z, w, h, d)
    
    placements = [None] * n
    
    for orig_idx, (bw, bh, bd) in box_list:
        best = None  # (waste, orientation, space_idx, x, y, z, w, h, d)
        for ox, oy, oz in [(bw, bh, bd), (bw, bd, bh), (bh, bw, bd), (bh, bd, bw), (bd, bw, bh), (bd, bh, bw)]:
            for si, (sx, sy, sz, sw, sh, sd) in enumerate(free_spaces):
                if ox <= sw and oy <= sh and oz <= sd:
                    # Calculate waste: volume of free space not used by this box
                    waste = sw * sh * sd - ox * oy * oz
                    # Prefer smaller waste, then smaller remaining dimensions
                    if best is None or (waste, sw - ox, sh - oy, sd - oz) < (best[0], best[1], best[2], best[3]):
                        best = (waste, (ox, oy, oz), si, sx, sy, sz, sw, sh, sd)
        
        if best is None:
            placements[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
            continue
        
        _, (ox, oy, oz), si, sx, sy, sz, sw, sh, sd = best
        
        # Place the box
        placements[orig_idx] = (0, sx, sy, sz, ox, oy, oz)
        
        # Remove the free space and add new ones
        del free_spaces[si]
        
        # Split the space:
        # 1. Remaining in x: (sx + ox, sy, sz, sw - ox, sh, sd) if sw - ox > 0
        # 2. Remaining in y: (sx, sy + oy, sz, ox, sh - oy, sd) if sh - oy > 0
        # 3. Remaining in z: (sx, sy, sz + oz, ox, oy, sd - oz) if sd - oz > 0
        
        if sw - ox > 0:
            free_spaces.append((sx + ox, sy, sz, sw - ox, sh, sd))
        if sh - oy > 0:
            free_spaces.append((sx, sy + oy, sz, ox, sh - oy, sd))
        if sd - oz > 0:
            free_spaces.append((sx, sy, sz + oz, ox, oy, sd - oz))
        
        # Merge overlapping free spaces (optional, for efficiency)
        # For now, skip merging to keep it simple.
    
    return placements

# EVOLVE-BLOCK-END
