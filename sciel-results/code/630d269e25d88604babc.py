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
        box_options.append((idx, valid_perms))
    
    # Sort by volume descending (Best-Fit Decreasing)
    box_options.sort(key=lambda x: x[1][0][0] * x[1][0][1] * x[1][0][2], reverse=True)
    
    # Track placed boxes with their occupied space
    placed = []  # list of (x, y, z, w, h, d)
    
    results = [None] * len(boxes)
    
    # For each box, try to find a position
    for idx, perms in box_options:
        found = False
        # Try each orientation
        for (bw, bh, bd) in perms:
            # Try to find a position using a simple scan
            # We'll use a grid-based approach or just try key positions
            # Key positions: (0,0,0), (x,0,0), (0,y,0), (0,0,z), etc.
            # Better: try all combinations of existing box faces + 0
            
            candidate_positions = set()
            candidate_positions.add((0, 0, 0))
            
            for (px, py, pz, pw, ph, pd) in placed:
                # Place adjacent to existing boxes
                candidate_positions.add((px + pw, py, pz))
                candidate_positions.add((px, py + ph, pz))
                candidate_positions.add((px, py, pz + pd))
            
            # Also try positions based on current "frontier"
            # Simple approach: try all candidate positions
            
            for (cx, cy, cz) in candidate_positions:
                # Check if box fits at this position
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check for overlap with all placed boxes
                overlap = False
                for (px, py, pz, pw, ph, pd) in placed:
                    # Check if intervals overlap in all 3 dimensions
                    if not (cx + bw <= px or px + pw <= cx):
                        if not (cy + bh <= py or py + ph <= cy):
                            if not (cz + bd <= pz or pz + pd <= cz):
                                overlap = True
                                break
                if overlap:
                    continue
                
                # Place the box
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
                found = True
                break
            
            if found:
                break
        
        if not found:
            # Box couldn't be placed
            results[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    return results
# EVOLVE-BLOCK-END
