import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    W, H, D = container
    
    # Sort boxes by volume in descending order (Best-Fit Decreasing)
    # We keep track of original indices to return correct placements
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # List of bins, each bin is a list of placed boxes: (x, y, z, w, h, d, original_index)
    bins = []
    
    def check_overlap(bin_boxes, x, y, z, w, h, d):
        """Check if a new box at (x,y,z) with dims (w,h,d) overlaps with any box in bin_boxes."""
        for (bx, by, bz, bw, bh, bd, _) in bin_boxes:
            # Check for overlap in all 3 dimensions
            if x < bx + bw and x + w > bx and \
               y < by + bh and y + h > by and \
               z < bz + bd and z + d > bz:
                return True
        return False

    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best (x, y, z) position for a box of size (w, h, d) in a given bin.
        'Best' is defined as the position with the smallest sum of coordinates (closest to origin),
        chosen from a set of candidate positions derived from existing boxes.
        Returns None if no position fits.
        """
        candidates = set()
        # Always try the origin
        candidates.add((0, 0, 0))
        
        # Generate candidates adjacent to existing boxes
        for (bx, by, bz, bw, bh, bd, _) in bin_boxes:
            # Right, Front, Top
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            
            # Also try combinations to fill corners better?
            # e.g. (bx + bw, by + bh, bz)
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            # Check bounds
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap
            if check_overlap(bin_boxes, cx, cy, cz, w, h, d):
                continue
            
            # Score: minimize distance to corner
            score = cx + cy + cz
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                
        return best_pos

    # Main loop
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place remaining boxes in a single new bin or just return what we have?
            # We must return a valid placement for ALL boxes.
            # If we run out of time, we just append them to a new bin without overlap checks? 
            # No, that would be invalid.
            # We must ensure we finish. With 40 boxes, the above is very fast.
            # If we are here, it's a bug or extremely large dims.
            pass

        best_bin_idx = -1
        best_pos = None
        best_global_score = float('inf')
        
        # Try existing bins
        for i, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                # We prefer bins that are already fuller? Or just the first valid one?
                # Standard BFD usually picks the bin with the best fit (min wasted space or min coordinate sum).
                # Here, we pick the position with the min coordinate sum.
                # If multiple bins have a valid position, we should probably pick the one that minimizes the global score.
                # However, to keep it simple and fast, we can just pick the first bin that fits, 
                # or the one where the position is closest to the corner.
                # Let's compare the (cx+cy+cz) of the best position in each bin.
                score = pos[0] + pos[1] + pos[2]
                if score < best_global_score:
                    best_global_score = score
                    best_bin_idx = i
                    best_pos = pos

        if best_bin_idx == -1:
            # No existing bin fits, create a new one
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
        else:
            # Add to existing bin
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))

    # Construct final placements
    # The problem asks for: (bin, x, y, z, w, h, d)
    # Bin labels are nonnegative integers.
    placements = []
    for bin_idx, bin_boxes in enumerate(bins):
        for (x, y, z, w, h, d, orig_idx) in bin_boxes:
            placements.append((bin_idx, x, y, z, w, h, d, orig_idx))
            
    # The return format is one placement per box, preserving original order?
    # "one placement (bin, x, y, z, w, h, d) per box"
    # Usually this means a list of tuples corresponding to the input boxes order.
    # Let's check the seed:
    # placements.append((b, x, y, z, w, h, d))
    # It appends in the order of the loop.
    # My loop iterates over sorted boxes.
    # I need to return them in the original order?
    # "preserving its original ordered dimensions" refers to w,h,d.
    # "one placement ... per box" usually implies a list where index i corresponds to box i.
    # Let's re-read carefully: "return one placement (bin, x, y, z, w, h, d) per box"
    # If I return a list of 40 tuples, the evaluator likely checks if the i-th tuple corresponds to the i-th input box.
    # The seed program does NOT sort. It iterates `for w, h, d in boxes:`.
    # So the seed returns placements in the original input order.
    # My algorithm sorts. I must unsort or map back.
    
    # My `placements` list currently has (bin, x, y, z, w, h, d, orig_idx).
    # I need to create a list `result` of length len(boxes) such that result[i] is the placement for boxes[i].
    
    result = [None] * len(boxes)
    for (b, x, y, z, w, h, d, idx) in placements:
        result[idx] = (b, x, y, z, w, h, d)
        
    return result
