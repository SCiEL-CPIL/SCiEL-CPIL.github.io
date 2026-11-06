"""3D Bin Packing: place boxes unrotated into numbered bins, minimize bins + last-bin fill penalty."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict

def pack(boxes, container):
    """
    3D Bin Packing with Best-Fit Decreasing and local improvement.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending, keep original index
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[1])
    
    # Data structures
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin without overlap and within bounds."""
        if x < 0 or y < 0 or z < 0:
            return False
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
    
    def bin_volume(bin_boxes):
        """Compute total volume of boxes in a bin."""
        return sum(w * h * d for (_, _, _, w, h, d) in bin_boxes)
    
    def compute_score(bins_list, placement_dict):
        """
        Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction).
        B is the number of occupied bins.
        """
        if not bins_list:
            return 0.0
        num_bins = len(bins_list)
        # Find the last bin (largest label)
        last_bin_idx = max(placement_dict.values(), key=lambda x: x[0]) if placement_dict else 0
        last_bin_idx = max(x[0] for x in placement_dict.values()) if placement_dict else 0
        last_bin_volume = bin_volume(bins_list[last_bin_idx]) if last_bin_idx < len(bins_list) else 0
        container_vol = W * H * D
        last_bin_frac = last_bin_volume / container_vol if container_vol > 0 else 0
        return num_bins + 0.1 * (1 - last_bin_frac)
    
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
    
    # Local improvement: try to reduce number of bins or improve last-bin fill
    # We'll do a limited number of improvement passes
    max_improvement_passes = 5
    for pass_num in range(max_improvement_passes):
        if time.time() > deadline - 20.0:
            break
        
        improved = False
        # Try to move boxes from the last bin to other bins
        last_bin_idx = max(x[0] for x in placement.values()) if placement else -1
        if last_bin_idx < 0 or last_bin_idx >= len(bins):
            break
        
        # Get boxes in the last bin
        last_bin_boxes = [(idx, x, y, z, w, h, d) 
                         for idx, (b_idx, x, y, z) in placement.items()
                         if b_idx == last_bin_idx]
        
        for box_info in last_bin_boxes:
            idx, bx, by, bz, bw, bh, bd = box_info
            # Try to move this box to another bin
            for target_bin in range(len(bins)):
                if target_bin == last_bin_idx:
                    continue
                # Remove box from last bin temporarily
                last_bin_copy = [b for b in bins[last_bin_idx] 
                                if not (b[0] == bx and b[1] == by and b[2] == bz)]
                # Try to place in target bin
                spot = find_best_spot(bins[target_bin], bw, bh, bd)
                if spot is not None:
                    # Check if this improves the score
                    # Simulate the move
                    old_score = compute_score(bins, placement)
                    
                    # Perform the move
                    bins[last_bin_idx] = last_bin_copy
                    tx, ty, tz = spot
                    bins[target_bin].append((tx, ty, tz, bw, bh, bd))
                    placement[idx] = (target_bin, tx, ty, tz)
                    
                    new_score = compute_score(bins, placement)
                    
                    if new_score < old_score - 1e-9:
                        improved = True
                        break
                    else:
                        # Revert the move
                        bins[target_bin] = [b for b in bins[target_bin] 
                                           if not (b[0] == tx and b[1] == ty and b[2] == tz)]
                        bins[last_bin_idx] = [b for b in bins[last_bin_idx] 
                                             if not (b[0] == bx and b[1] == by and b[2] == bz)]
                        bins[last_bin_idx].append((bx, by, bz, bw, bh, bd))
                        placement[idx] = (last_bin_idx, bx, by, bz)
            
            if improved:
                break
        
        if not improved:
            break
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
# EVOLVE-BLOCK-END
