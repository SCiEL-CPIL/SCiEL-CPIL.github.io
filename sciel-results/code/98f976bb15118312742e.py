"""Improved program for 3D Container Loading using Best-Fit Decreasing Height with rotation."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Generate all 6 permutations of dimensions for each box
    box_options = []
    for idx, (w, h, d) in enumerate(boxes):
        perms = set()
        perms.add((w, h, d))
        perms.add((w, d, h))
        perms.add((h, w, d))
        perms.add((h, d, w))
        perms.add((d, w, h))
        perms.add((d, h, w))
        # Only keep valid orientations that fit in container
        valid_perms = []
        for p in perms:
            if p[0] <= W and p[1] <= H and p[2] <= D:
                valid_perms.append(p)
        if not valid_perms:
            # If no orientation fits, keep original for marking as left out
            valid_perms = [(w, h, d)]
        
        # Precompute min dimensions for early pruning
        min_w = min(p[0] for p in valid_perms)
        min_h = min(p[1] for p in valid_perms)
        min_d = min(p[2] for p in valid_perms)
        
        box_options.append((idx, valid_perms, min_w, min_h, min_d))
    
    # Sort by volume descending (Best-Fit Decreasing)
    # Using the volume of the first valid permutation as a proxy, or max volume?
    # Volume is invariant under rotation.
    box_options.sort(key=lambda x: x[1][0][0] * x[1][0][1] * x[1][0][2], reverse=True)
    
    # Track placed boxes with their occupied space
    placed = []  # list of (x, y, z, w, h, d)
    
    results = [None] * len(boxes)
    
    # For each box, try to find a position
    for idx, perms, min_w, min_h, min_d in box_options:
        found = False
        
        # Generate candidate coordinates based on placed boxes and container bounds
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_coords.add(px)
            x_coords.add(px + pw)
            y_coords.add(py)
            y_coords.add(py + ph)
            z_coords.add(pz)
            z_coords.add(pz + pd)
            
        # Sort coordinates to prioritize lower values (bottom, left, front)
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        z_coords = sorted(z_coords)
        
        # Iterate through candidates in priority order
        for cz in z_coords:
            # If the box's minimum depth doesn't fit in the remaining space from this z
            if cz + min_d > D:
                break
            
            for cy in y_coords:
                # If the box's minimum height doesn't fit in the remaining space from this y
                if cy + min_h > H:
                    break
                
                for cx in x_coords:
                    # If the box's minimum width doesn't fit in the remaining space from this x
                    if cx + min_w > W:
                        break
                    
                    # Try each orientation to see if it fits at (cx, cy, cz)
                    valid_orientation = None
                    for (bw, bh, bd) in perms:
                        # Check container bounds
                        if cx + bw > W or cy + bh > H or cz + bd > D:
                            continue
                        
                        # Check overlap with placed boxes
                        overlap = False
                        for (px, py, pz, pw, ph, pd) in placed:
                            # Check if intervals overlap in all 3 dimensions
                            if not (cx + bw <= px or px + pw <= cx):
                                if not (cy + bh <= py or py + ph <= cy):
                                    if not (cz + bd <= pz or pz + pd <= cz):
                                        overlap = True
                                        break
                        if not overlap:
                            valid_orientation = (bw, bh, bd)
                            break
                    
                    if valid_orientation:
                        bw, bh, bd = valid_orientation
                        placed.append((cx, cy, cz, bw, bh, bd))
                        results[idx] = (0, cx, cy, cz, bw, bh, bd)
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
        
        if not found:
            # Box couldn't be placed
            results[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    return results
# EVOLVE-BLOCK-END
