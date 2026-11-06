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
    
    # Free Space Representation: List of tuples (x, y, z, w, h, d)
    # Each tuple represents a rectangular prism of empty space.
    free_spaces = [(0, 0, 0, W, H, D)]
    
    # Helper to split a free space given a box placed at (px, py, pz) with dims (ow, oh, od)
    # The box is assumed to fit inside the free space (sx, sy, sz, sw, sh, sd)
    def split_space(sx, sy, sz, sw, sh, sd, px, py, pz, ow, oh, od):
        new_spaces = []
        
        # 1. Left/Right split (X-axis)
        if px > sx:
            new_spaces.append((sx, sy, sz, px - sx, sh, sd))
        if px + ow < sx + sw:
            new_spaces.append((px + ow, sy, sz, sx + sw - (px + ow), sh, sd))
        
        # 2. Front/Back split (Y-axis) - Only for the middle strip in X
        # The region between the left and right splits is [px, px+ow] in X
        # Within this X-strip, we split by Y
        if py > sy:
            new_spaces.append((px, sy, sz, ow, py - sy, sd))
        if py + oh < sy + sh:
            new_spaces.append((px, py + oh, sz, ow, sy + sh - (py + oh), sd))
            
        # 3. Up/Down split (Z-axis) - Only for the middle strip in X and Y
        # The region between the Y splits is [py, py+oh] in Y
        # Within this X-Y strip, we split by Z
        if pz > sz:
            new_spaces.append((px, py, sz, ow, oh, pz - sz))
        if pz + od < sz + sd:
            new_spaces.append((px, py, pz + od, ow, oh, sz + sd - (pz + od)))
            
        return new_spaces

    # 2. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        best_space_idx = -1
        
        # Iterate through all current free spaces
        for fs_idx, (sx, sy, sz, sw, sh, sd) in enumerate(free_spaces):
            if time.time() > deadline:
                break
                
            # Try all orientations
            for ow, oh, od in orients:
                # Check if box fits in this free space
                if ow > sw or oh > sh or od > sd:
                    continue
                
                # We can place the box anywhere within the free space.
                # To maximize packing density and keep the structure simple, 
                # we typically align the box to one of the corners of the free space.
                # However, to find the "best" position, we can just pick the 
                # corner that minimizes (z, y, x) relative to the free space origin,
                # or simply place it at the "bottom-left-front" of the free space 
                # relative to the global coordinates to encourage stacking.
                
                # Actually, to be more effective, we should try placing it at 
                # the corner of the free space that is closest to the origin (0,0,0) 
                # or simply the corner (sx, sy, sz) which is the "lowest" corner.
                # Since the free space is a rectangle, placing at (sx, sy, sz) 
                # is a valid and often good choice. 
                # However, sometimes placing it "higher" might allow better 
                # packing of subsequent boxes? No, FFD usually benefits from 
                # filling the lowest available space first.
                
                # Let's try placing at the corner (sx, sy, sz).
                px, py, pz = sx, sy, sz
                
                # Calculate score: prefer lower Z, then Y, then X
                # We want to minimize this score
                score = pz * 1000000 + py * 10000 + px
                
                # Optional: Tie-breaker. If scores are close, prefer the placement 
                # that leaves the largest remaining free space volume?
                # Or prefer the one that fits in the smallest free space (Best Fit)?
                # Best Fit is often better for packing.
                # Let's add a small penalty for the volume of the free space to encourage 
                # using smaller holes.
                # score += (sw * sh * sd) * 0.001
                
                if score < best_score:
                    best_score = score
                    best_placement = (px, py, pz, ow, oh, od)
                    best_space_idx = fs_idx
                    
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            
            # Update Free Spaces
            # Remove the used space
            sx, sy, sz, sw, sh, sd = free_spaces.pop(best_space_idx)
            
            # Add new sub-spaces
            new_spaces = split_space(sx, sy, sz, sw, sh, sd, px, py, pz, ow, oh, od)
            free_spaces.extend(new_spaces)
            
            # Prune empty or invalid spaces (though split_space should handle this)
            # It's good practice to filter out any with 0 volume just in case
            free_spaces = [fs for fs in free_spaces if fs[3] > 0 and fs[4] > 0 and fs[5] > 0]
            
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
