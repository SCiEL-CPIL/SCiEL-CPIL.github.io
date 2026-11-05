"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    
    def get_corners(bin_boxes):
        """Generate candidate corner positions for a bin."""
        if not bin_boxes:
            return [(0, 0, 0)]
        
        corners = set()
        corners.add((0, 0, 0))
        
        for (x, y, z, w, h, d, _) in bin_boxes:
            corners.add((x + w, y, z))
            corners.add((x, y + h, z))
            corners.add((x, y, z + d))
            corners.add((x + w, y + h, z))
            corners.add((x + w, y, z + d))
            corners.add((x, y + h, z + d))
            corners.add((x + w, y + h, z + d))
        
        return list(corners)
    
    def fits_at(bin_boxes, x, y, z, w, h, d):
        """Check if a box of size (w,h,d) fits at position (x,y,z) in bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd, _) in bin_boxes:
            if x < bx + bw and x + w > bx and \
               y < by + bh and y + h > by and \
               z < bz + bd and z + d > bz:
                return False
        return True
    
    def best_position(bin_boxes, w, h, d):
        """Find the best position for a box in a given bin. Returns (x,y,z) or None."""
        corners = get_corners(bin_boxes)
        best = None
        best_key = None
        for (x, y, z) in corners:
            if fits_at(bin_boxes, x, y, z, w, h, d):
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (x, y, z)
        return best
    
    # Place each box
    box_to_placement = {}
    
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin_idx = None
        best_pos = None
        best_score = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                score = (z, y, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx is None:
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin_idx].append((x, y, z, w, h, d, i))
        box_to_placement[i] = (best_bin_idx, x, y, z)
    
    # Build placements in original order
    placements = []
    for i, (w, h, d) in enumerate(boxes):
        if i in box_to_placement:
            b_idx, x, y, z = box_to_placement[i]
            placements.append((b_idx, x, y, z, w, h, d))
        else:
            # Safety net: place in a new bin at origin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, i)])
            placements.append((b_idx, 0, 0, 0, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
