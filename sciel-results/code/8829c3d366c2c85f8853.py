import time
import itertools

def load(boxes, container):
    start_time = time.time()
    # Set a safety deadline slightly below the 120s limit
    deadline = start_time + 115.0
    
    W, H, D = container
    
    # Store original dimensions and index
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Sort by volume descending to place large boxes first
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed = []
    
    # Initialize results with "left out" status (-1)
    # Format: (bin, x, y, z, w, h, d)
    results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    for idx, w, h, d in indexed_boxes:
        # Check time limit
        if time.time() > deadline:
            break
        
        # Quick check: does the box fit in the container in any rotation?
        fits_container = False
        rotations = list(set(itertools.permutations((w, h, d))))
        for rw, rh, rd in rotations:
            if rw <= W and rh <= H and rd <= D:
                fits_container = True
                break
        
        if not fits_container:
            continue
            
        placed_box = None
        
        # Iterate positions in order of preference:
        # 1. Lowest Z (bottom)
        # 2. Lowest Y (back)
        # 3. Lowest X (left)
        # This is a "First Fit" strategy on a dense grid.
        
        for z in range(D):
            if time.time() > deadline:
                break
            if placed_box is not None:
                break
            for y in range(H):
                if time.time() > deadline:
                    break
                if placed_box is not None:
                    break
                for x in range(W):
                    if time.time() > deadline:
                        break
                    if placed_box is not None:
                        break
                    
                    # Try each rotation at this (x, y, z)
                    for rw, rh, rd in rotations:
                        # Check bounds
                        if x + rw > W or y + rh > H or z + rd > D:
                            continue
                        
                        # Check overlap with placed boxes
                        valid = True
                        for px, py, pz, pw, ph, pd in placed:
                            # Check intersection
                            if (x < px + pw and px < x + rw and
                                y < py + ph and py < y + rh and
                                z < pz + pd and pz < z + rd):
                                valid = False
                                break
                        
                        if valid:
                            placed_box = (x, y, z, rw, rh, rd)
                            break
        
        if placed_box is not None:
            bx, by, bz, bw, bh, bd = placed_box
            placed.append(placed_box)
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
    return results
