"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using FFD with free-space tracking and local search.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Sort boxes by volume descending for FFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin will have a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists

    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def find_best_position(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin using comprehensive candidate generation.
        Returns (x, y, z) or None if no position found.
        """
        # Generate candidate positions from all "upper" corners of existing boxes
        candidates = {(0, 0, 0)}
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # All 8 corners that could be "lower-left-front" of new box
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        best_pos = None
        best_score = None
        
        for (xc, yc, zc) in candidates:
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            if can_place(bin_placements, xc, yc, zc, w, h, d):
                # Score: prefer positions that minimize the "bounding box" extent
                # Weighted sum to encourage compact packing
                score = xc * 1.0 + yc * 1.0 + zc * 1.0
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    def find_position_shelf(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin using shelf-based placement.
        Returns (x, y, z) or None if no position found.
        """
        # Collect all existing shelf levels (y positions)
        shelf_levels = set([0])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            shelf_levels.add(py + ph)
        
        best_pos = None
        best_score = None
        
        for y_level in sorted(shelf_levels):
            if y_level + h > H:
                continue
            # Find the best x position on this shelf
            x_candidates = set([0])
            for (px, py, pz, pw, ph, pd) in bin_placements:
                if py == y_level:
                    x_candidates.add(px + pw)
            
            for x in sorted(x_candidates):
                if x + w > W:
                    continue
                # Find the best z position
                z_candidates = set([0])
                for (px, py, pz, pw, ph, pd) in bin_placements:
                    if py == y_level and px <= x and px + pw >= x + w:
                        z_candidates.add(pz + pd)
                
                for z in sorted(z_candidates):
                    if z + d > D:
                        continue
                    if can_place(bin_placements, x, y_level, z, w, h, d):
                        score = x + y_level + z
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y_level, z)
                        break
        
        return best_pos

    def find_position(bin_placements, w, h, d):
        """
        Try multiple placement strategies and return the best position.
        """
        # Try corner-based
        pos_corner = find_best_position(bin_placements, w, h, d)
        if pos_corner is not None:
            return pos_corner
        
        # Try shelf-based
        pos_shelf = find_position_shelf(bin_placements, w, h, d)
        if pos_shelf is not None:
            return pos_shelf
        
        return None

    def bin_occupied_volume(bin_placements):
        """Calculate the total volume of boxes in a bin."""
        return sum(w * h * d for (x, y, z, w, h, d) in bin_placements)

    def bin_max_extent(bin_placements):
        """Calculate the max extent (bounding box) of boxes in a bin."""
        if not bin_placements:
            return (0, 0, 0)
        max_x = max(x + w for (x, y, z, w, h, d) in bin_placements)
        max_y = max(y + h for (x, y, z, w, h, d) in bin_placements)
        max_z = max(z + d for (x, y, z, w, h, d) in bin_placements)
        return (max_x, max_y, max_z)

    # Main packing loop - FFD
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins, preferring the bin where it fits most compactly
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize the resulting bounding box
                # Use sum of max extents as proxy for compactness
                new_max_x = max(x + w, *(px + pw for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else x + w
                new_max_y = max(y + h, *(py + ph for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else y + h
                new_max_z = max(z + d, *(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else z + d
                fit_score = new_max_x + new_max_y + new_max_z
                
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_bin = []
            x, y, z = 0, 0, 0
            if can_place(new_bin, x, y, z, w, h, d):
                new_bin.append((x, y, z, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, x, y, z, w, h, d)
            else:
                # This shouldn't happen if container is large enough, but handle gracefully
                new_bin.append((0, 0, 0, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # If we hit the deadline, place remaining boxes using a simple fallback
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        # Find any bin where it can be placed, or create a new one
        placed = False
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Local search: try to move boxes to reduce number of bins
    # For each bin from the last one, try to move boxes to earlier bins
    if time.time() < deadline - 20:  # Only if we have time
        # Try to merge bins by moving boxes
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            # Iterate from last bin to first
            for b_idx in range(len(bins) - 1, -1, -1):
                if time.time() > deadline - 20:
                    break
                if not bins[b_idx]:
                    continue
                # Try to move each box in this bin to an earlier bin
                boxes_to_try = list(bins[b_idx])
                for box in boxes_to_try:
                    x, y, z, w, h, d = box
                    # Try to find a position in an earlier bin
                    moved = False
                    for target_bin in range(b_idx):
                        if time.time() > deadline - 20:
                            break
                        pos = find_position(bins[target_bin], w, h, d)
                        if pos is not None:
                            # Move the box
                            bins[b_idx].remove(box)
                            new_x, new_y, new_z = pos
                            bins[target_bin].append((new_x, new_y, new_z, w, h, d))
                            # Update result
                            for i, (bin_id, bx, by, bz, bw, bh, bd) in enumerate(result):
                                if (bx, by, bz, bw, bh, bd) == (x, y, z, w, h, d) and bin_id == b_idx:
                                    result[i] = (target_bin, new_x, new_y, new_z, w, h, d)
                                    break
                            moved = True
                            changed = True
                            break
                    if not bins[b_idx]:
                        # Bin is now empty, remove it
                        bins.pop(b_idx)
                        # Renumber bins
                        for i in range(len(result)):
                            bin_id = result[i][0]
                            if bin_id > b_idx:
                                result[i] = (bin_id - 1,) + result[i][1:]
                            elif bin_id == b_idx:
                                # This shouldn't happen since bin is empty
                                pass
                        break

    return result
# EVOLVE-BLOCK-END
