"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pack(boxes, container):
    """
    Best-Fit Decreasing 3D bin packing with candidate-position enumeration.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending, keep original index
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[1])
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    # Data structures
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_spot(bin_boxes, w, h, d):
        """
        Find the best (x, y, z) position for a box in a given bin.
        Uses candidate positions derived from box faces.
        Returns (x, y, z) or None if no valid position found.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate coordinate sets
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            x_vals.add(px + pw)
            y_vals.add(py + ph)
            z_vals.add(pz + pd)
        
        # Filter to valid starting positions
        x_vals = sorted([x for x in x_vals if x + w <= W])
        y_vals = sorted([y for y in y_vals if y + h <= H])
        z_vals = sorted([z for z in z_vals if z + d <= D])
        
        if not x_vals or not y_vals or not z_vals:
            return None
        
        # Try positions in order of compactness (minimize x+y+z)
        # Sort all candidate combinations by sum, then try them
        best_pos = None
        best_cost = float('inf')
        
        for z in z_vals:
            for y in y_vals:
                for x in x_vals:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        cost = x + y + z
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Process boxes in decreasing volume order
    for idx, vol, w, h, d in indexed_boxes:
        # Time check
        if time.time() > deadline:
            # Fallback: place remaining boxes with simple shelf heuristic
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                if idx2 in placement:
                    continue
                placed = False
                # Try existing bins with simple origin check
                for b_idx in range(len(bins)):
                    if can_place(bins[b_idx], 0, 0, 0, w2, h2, d2):
                        bins[b_idx].append((0, 0, 0, w2, h2, d2))
                        placement[idx2] = (b_idx, 0, 0, 0)
                        placed = True
                        break
                if not placed:
                    # Try to find any spot in existing bins
                    for b_idx in range(len(bins)):
                        spot = find_best_spot(bins[b_idx], w2, h2, d2)
                        if spot is not None:
                            x, y, z = spot
                            bins[b_idx].append((x, y, z, w2, h2, d2))
                            placement[idx2] = (b_idx, x, y, z)
                            placed = True
                            break
                if not placed:
                    # Create new bin
                    bins.append([(0, 0, 0, w2, h2, d2)])
                    placement[idx2] = (len(bins) - 1, 0, 0, 0)
            
            # Build result
            result = []
            for i in range(n):
                b_idx, x, y, z = placement[i]
                w_i, h_i, d_i = boxes[i]
                result.append((b_idx, x, y, z, w_i, h_i, d_i))
            return result
        
        # Try to place in existing bins using best-fit
        best_bin = -1
        best_x, best_y, best_z = 0, 0, 0
        best_cost = float('inf')
        
        for b_idx in range(len(bins)):
            spot = find_best_spot(bins[b_idx], w, h, d)
            if spot is not None:
                x, y, z = spot
                cost = x + y + z
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b_idx
                    best_x, best_y, best_z = x, y, z
        
        if best_bin == -1:
            # Create new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_x, best_y, best_z = 0, 0, 0
        
        # Place the box
        bins[best_bin].append((best_x, best_y, best_z, w, h, d))
        placement[idx] = (best_bin, best_x, best_y, best_z)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
# EVOLVE-BLOCK-END
