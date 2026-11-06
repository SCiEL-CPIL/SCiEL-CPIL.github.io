"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    # Create list of (index, volume, w, h, d)
    indexed_boxes = [(i, w*h*d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    # Sort by volume descending
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Result storage: index -> (bin, x, y, z, w, h, d)
    result = [None] * n
    
    # Bins: list of lists of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place_in_bin(bin_boxes, x, y, z, w, h, d):
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Check overlap
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True

    def find_spot(bin_boxes, w, h, d):
        # Try to find a spot in this bin.
        # Heuristic: try points defined by (0,0,0) and faces of existing boxes.
        # This is a standard "skyline" or "frontier" approach simplified.
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
            
        # Filter candidates that are within bounds
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        # Iterate in order of preference (lower is better for compactness)
        for z in zs:
            for y in ys:
                for x in xs:
                    if can_place_in_bin(bin_boxes, x, y, z, w, h, d):
                        return (x, y, z)
        return None

    start_time = time.time()
    deadline = start_time + 140 # 10s margin for safety
    
    for i, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # This is unlikely for 40 boxes, but safe.
            break
            
        placed = False
        
        # Try to place in existing bins (First Fit)
        for b_idx in range(len(bins)):
            spot = find_spot(bins[b_idx], w, h, d)
            if spot is not None:
                x, y, z = spot
                bins[b_idx].append((x, y, z, w, h, d))
                result[i] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Open new bin
            new_bin = []
            spot = find_spot(new_bin, w, h, d)
            if spot is not None:
                x, y, z = spot
                new_bin.append((x, y, z, w, h, d))
                result[i] = (len(bins), x, y, z, w, h, d)
                bins.append(new_bin)
                placed = True
            else:
                # Fallback: place at 0,0,0 (should fit if box is valid)
                new_bin.append((0, 0, 0, w, h, d))
                result[i] = (len(bins), 0, 0, 0, w, h, d)
                bins.append(new_bin)
                placed = True
                
    # Ensure all boxes are assigned a placement
    for i in range(n):
        if result[i] is None:
            # Emergency fallback: assign to a new bin at 0,0,0
            # This ensures the function returns a complete list, though it might be invalid if overlaps occur.
            # However, the main loop should handle all valid boxes.
            result[i] = (1000, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result
# EVOLVE-BLOCK-END
