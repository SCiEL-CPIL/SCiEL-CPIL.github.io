"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
import time

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 possible rotations for each box
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in permutations([w, h, d]):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Precompute rotations for all boxes
    box_rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Helper to check if a box fits in the container
    def fits_in_container(px, py, pz, pw, ph, pd):
        return (px >= 0 and py >= 0 and pz >= 0 and 
                px + pw <= W and py + ph <= H and pz + pd <= D)

    # Helper to check overlap
    def has_overlap(px, py, pz, pw, ph, pd, placed_boxes):
        for (ex, ey, ez, ew, eh, ed) in placed_boxes:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return True
        return False

    # New Heuristic: Bottom-Left-Back (BLB)
    # Score: Minimize x + y + z (distance to origin)
    # This pushes boxes to the corner, creating a tight pack.
    
    def find_best_placement(idx, placed_boxes):
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        for (rw, rh, rd) in box_rotations[idx]:
            # Generate candidate points: (0,0,0) and corners of existing boxes
            candidates = set()
            candidates.add((0, 0, 0))
            for (ex, ey, ez, ew, eh, ed) in placed_boxes:
                for dx in [0, ew]:
                    for dy in [0, eh]:
                        for dz in [0, ed]:
                            candidates.add((ex + dx, ey + dy, ez + dz))
            
            for (px, py, pz) in candidates:
                if not fits_in_container(px, py, pz, rw, rh, rd):
                    continue
                if has_overlap(px, py, pz, rw, rh, rd, placed_boxes):
                    continue
                
                # Score: Minimize distance to origin
                score = px + py + pz
                
                # Tie-breaker: Prefer lower Z, then Y, then X
                # This helps in stacking
                score += pz * 0.001 + py * 0.0001 + px * 0.00001
                
                if score < best_score:
                    best_score = score
                    best_pos = (px, py, pz)
                    best_rot = (rw, rh, rd)
                    # If we found the origin, it's the best possible position
                    if px == 0 and py == 0 and pz == 0:
                        break
            
            # If we found the origin, no need to check other rotations
            if best_pos is not None and best_pos[0] == 0 and best_pos[1] == 0 and best_pos[2] == 0:
                break
        
        return best_pos, best_rot

    # Define multiple sorting strategies to try
    # 1. Volume Descending
    # 2. Max Dimension Descending
    # 3. Surface Area Descending
    # 4. Sum of Dimensions Descending
    
    strategies = [
        sorted(range(n), key=lambda i: box_volumes[i], reverse=True),
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),
        sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True),
        sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True),
    ]
    
    best_total_volume = 0
    best_placements = None
    
    start_time = time.time()
    
    for order in strategies:
        if time.time() - start_time > 100: # Leave some time for safety
            break
            
        placements = [None] * n
        placed_boxes = []
        total_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            pos, rot = find_best_placement(idx, placed_boxes)
            
            if pos is not None:
                px, py, pz = pos
                rw, rh, rd = rot
                placements[idx] = (0, px, py, pz, rw, rh, rd)
                placed_boxes.append((px, py, pz, rw, rh, rd))
                total_vol += box_volumes[idx]
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if total_vol > best_total_volume:
            best_total_volume = total_vol
            best_placements = placements[:]
            
    # Fill any remaining None with -1
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    else:
        for i in range(n):
            if best_placements[i] is None:
                w, h, d = boxes[i]
                best_placements[i] = (-1, 0, 0, 0, w, h, d)
                
    return best_placements
# EVOLVE-BLOCK-END
