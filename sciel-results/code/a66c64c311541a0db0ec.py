"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending (First-Fit-Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) in the given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check overlap
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True
    
    def find_spot(bin_boxes, w, h, d):
        """Find a valid position in the bin for a box of size (w,h,d).
        Uses a frontier-based approach: try positions based on existing boxes' boundaries."""
        # Candidate x positions: 0 and right edges of existing boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_candidates.add(bx + bw)
            y_candidates.add(by + bh)
            z_candidates.add(bz + bd)
        
        # Filter candidates to valid ranges
        x_candidates = sorted([x for x in x_candidates if x + w <= W])
        y_candidates = sorted([y for y in y_candidates if y + h <= H])
        z_candidates = sorted([z for z in z_candidates if z + d <= D])
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        
        # Try all combinations, preferring positions closer to origin (bottom-left-front)
        # Sort by (z, y, x) to prefer filling from bottom
        best_pos = None
        best_key = None
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
                        if best_key == (0, 0, 0):
                            return best_pos
        return best_pos
    
    # Pack each box
    for idx, w, h, d, vol in indexed_boxes:
        placed = False
        
        # Try to place in existing bins (first-fit)
        for b_idx, bin_boxes in enumerate(bins):
            spot = find_spot(bin_boxes, w, h, d)
            if spot is not None:
                bin_boxes.append((spot[0], spot[1], spot[2], w, h, d))
                placed = True
                break
        
        # If not placed, open a new bin
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
        
        # Check deadline
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple way
            # This should rarely happen, but ensure validity
            for remaining in indexed_boxes[indexed_boxes.index((idx, w, h, d, vol)) + 1:]:
                ridx, rw, rh, rd, rvol = remaining
                # Try existing bins
                placed_rem = False
                for b_idx, bin_boxes in enumerate(bins):
                    spot = find_spot(bin_boxes, rw, rh, rd)
                    if spot is not None:
                        bin_boxes.append((spot[0], spot[1], spot[2], rw, rh, rd))
                        placed_rem = True
                        break
                if not placed_rem:
                    bins.append([(0, 0, 0, rw, rh, rd)])
            break
    
    # Build result: map back to original box order
    # We need to track which original index went where
    # Let's redo with proper tracking
    
    # Actually, let me restructure to track original indices properly
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    bins = []
    
    def can_place2(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True
    
    def find_spot2(bin_boxes, w, h, d):
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_candidates.add(bx + bw)
            y_candidates.add(by + bh)
            z_candidates.add(bz + bd)
        x_candidates = sorted([x for x in x_candidates if x + w <= W])
        y_candidates = sorted([y for y in y_candidates if y + h <= H])
        z_candidates = sorted([z for z in z_candidates if z + d <= D])
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        best_pos = None
        best_key = None
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place2(bin_boxes, x, y, z, w, h, d):
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
                        if best_key == (0, 0, 0):
                            return best_pos
        return best_pos
    
    # Track placement: original_index -> (bin_idx, x, y, z)
    placement_map = {}
    
    for idx, w, h, d, vol in indexed_boxes:
        placed = False
        for b_idx, bin_boxes in enumerate(bins):
            spot = find_spot2(bin_boxes, w, h, d)
            if spot is not None:
                bin_boxes.append((spot[0], spot[1], spot[2], w, h, d))
                placement_map[idx] = (b_idx, spot[0], spot[1], spot[2])
                placed = True
                break
        if not placed:
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[idx] = (new_bin_idx, 0, 0, 0)
        
        if time.time() > deadline:
            # Place remaining in new bins to ensure validity
            for remaining in indexed_boxes[indexed_boxes.index((idx, w, h, d, vol)) + 1:]:
                ridx, rw, rh, rd, rvol = remaining
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, rw, rh, rd)])
                placement_map[ridx] = (new_bin_idx, 0, 0, 0)
            break
    
    # Build final placements in original order
    result = []
    for i in range(n):
        if i in placement_map:
            b_idx, x, y, z = placement_map[i]
            w, h, d = boxes[i]
            result.append((b_idx, x, y, z, w, h, d))
        else:
            # Should not happen, but safety
            result.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
    
    return result

# EVOLVE-BLOCK-END
