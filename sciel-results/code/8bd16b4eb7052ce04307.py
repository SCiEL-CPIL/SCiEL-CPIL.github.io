"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using vectorized NumPy collision checks.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # safety margin before 120s kill
    container_vol = W * H * D

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin: [placements_list, occupancy_grid]
    bins = []

    def make_grid():
        return np.zeros((W, H, D), dtype=np.bool_)

    def place_in_grid(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = True

    def find_position_corner(bin_data, w, h, d):
        """
        Find a valid position using corner-based candidate generation.
        Returns (x, y, z) or None.
        """
        placements, grid = bin_data
        
        # Generate candidate positions from corners of existing boxes
        cand_list = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placements:
            cand_list.append((px + pw, py, pz))
            cand_list.append((px, py + ph, pz))
            cand_list.append((px, py, pz + pd))
            cand_list.append((px + pw, py + ph, pz))
            cand_list.append((px + pw, py, pz + pd))
            cand_list.append((px, py + ph, pz + pd))
            cand_list.append((px + pw, py + ph, pz + pd))
        
        # Deduplicate and filter by bounds
        seen = set()
        valid_cands = []
        for (xc, yc, zc) in cand_list:
            if (xc, yc, zc) in seen:
                continue
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            seen.add((xc, yc, zc))
            valid_cands.append((xc, yc, zc))
        
        if not valid_cands:
            return None
        
        best_pos = None
        best_score = float('inf')
        
        for (xc, yc, zc) in valid_cands:
            if not np.any(grid[xc:xc+w, yc:yc+h, zc:zc+d]):
                score = xc * 10000 + yc * 100 + zc
                if score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    def find_position_exhaustive(bin_data, w, h, d):
        """
        Exhaustive search over all valid positions using vectorized checks.
        Used when corner-based approach fails.
        """
        placements, grid = bin_data
        
        # Build 3D prefix sum
        grid_int = grid.astype(np.int32)
        padded = np.zeros((W+1, H+1, D+1), dtype=np.int32)
        padded[1:, 1:, 1:] = grid_int
        prefix = np.cumsum(np.cumsum(np.cumsum(padded, axis=0), axis=1), axis=2)
        
        best_pos = None
        best_score = float('inf')
        
        for zc in range(D - d + 1):
            a = prefix[w:, h:, zc+d]
            b = prefix[:W-w+1, h:, zc+d]
            c = prefix[w:, :H-h+1, zc+d]
            d_slice = prefix[w:, h:, zc]
            e = prefix[:W-w+1, :H-h+1, zc+d]
            f = prefix[:W-w+1, h:, zc]
            g = prefix[w:, :H-h+1, zc]
            h_slice = prefix[:W-w+1, :H-h+1, zc]
            
            sub_sums = a - b - c - d_slice + e + f + g - h_slice
            
            valid_mask = (sub_sums == 0)
            if np.any(valid_mask):
                valid_xs, valid_ys = np.where(valid_mask)
                scores = valid_xs + valid_ys
                min_idx = np.argmin(scores)
                x_best = valid_xs[min_idx]
                y_best = valid_ys[min_idx]
                score = x_best * 10000 + y_best * 100 + zc
                if score < best_score:
                    best_score = score
                    best_pos = (int(x_best), int(y_best), int(zc))
        
        return best_pos

    def try_place(bin_data, w, h, d):
        """Try to place a box in a bin. Returns (x,y,z) or None."""
        pos = find_position_corner(bin_data, w, h, d)
        if pos is None:
            pos = find_position_exhaustive(bin_data, w, h, d)
        return pos

    # Main packing loop
    result = [None] * n
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_data in enumerate(bins):
            pos = try_place(bin_data, w, h, d)
            if pos is not None:
                x, y, z = pos
                fit_score = (max(x + w, y + h, z + d), x + w + y + h + z + d)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin][0].append((x, y, z, w, h, d))
            place_in_grid(bins[best_bin][1], x, y, z, w, h, d)
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            new_grid = make_grid()
            new_bin = [[(0, 0, 0, w, h, d)], new_grid]
            place_in_grid(new_grid, 0, 0, 0, w, h, d)
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Fallback for any unplaced boxes
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        placed = False
        for b_idx, bin_data in enumerate(bins):
            pos = try_place(bin_data, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx][0].append((x, y, z, w, h, d))
                place_in_grid(bins[b_idx][1], x, y, z, w, h, d)
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_grid = make_grid()
            new_bin = [[(0, 0, 0, w, h, d)], new_grid]
            place_in_grid(new_grid, 0, 0, 0, w, h, d)
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Local search: try to reduce bin count by moving boxes
    # Strategy: for each bin from the last, try to move its boxes to earlier bins
    # This can potentially eliminate empty bins
    
    # First, identify which bins are actually occupied
    bin_occupied = [False] * len(bins)
    for i, (b, x, y, z, w, h, d) in enumerate(result):
        if b is not None:
            bin_occupied[b] = True
    
    # Try to consolidate: for each box in later bins, try moving to earlier bins
    # Process bins from highest to lowest, trying to move boxes out
    for b_idx in range(len(bins) - 1, -1, -1):
        if time.time() > deadline:
            break
        if not bin_occupied[b_idx]:
            continue
        
        # Get all boxes in this bin
        boxes_in_bin = [(i, result[i]) for i in range(n) if result[i] is not None and result[i][0] == b_idx]
        
        # Try to move each box to an earlier bin
        for i, (box_idx, placement) in enumerate(boxes_in_bin):
            if time.time() > deadline:
                break
            _, bx, by, bz, bw, bh, bd = placement
            
            # Temporarily remove from current bin
            # Find and remove from bins[b_idx][0]
            bins[b_idx][0].remove((bx, by, bz, bw, bh, bd))
            place_in_grid(bins[b_idx][1], bx, by, bz, bw, bh, bd)
            # Note: we need to unplace, so set to False
            bins[b_idx][1][bx:bx+bw, by:by+bh, bz:bz+bd] = False
            
            # Try to place in earlier bins
            placed_elsewhere = False
            for target_bin in range(b_idx):
                if not bin_occupied[target_bin] and target_bin != b_idx:
                    # If target bin is empty, we can place there directly
                    pos = try_place(bins[target_bin], bw, bh, bd)
                    if pos is not None:
                        tx, ty, tz = pos
                        bins[target_bin][0].append((tx, ty, tz, bw, bh, bd))
                        place_in_grid(bins[target_bin][1], tx, ty, tz, bw, bh, bd)
                        result[box_idx] = (target_bin, tx, ty, tz, bw, bh, bd)
                        bin_occupied[target_bin] = True
                        placed_elsewhere = True
                        break
            
            if not placed_elsewhere:
                # Put it back
                bins[b_idx][0].append((bx, by, bz, bw, bh, bd))
                place_in_grid(bins[b_idx][1], bx, by, bz, bw, bh, bd)
        
        # Check if bin is now empty
        if not bins[b_idx][0]:
            bin_occupied[b_idx] = False

    # Final pass: ensure all boxes are placed
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is None:
            placed = False
            for b_idx, bin_data in enumerate(bins):
                pos = try_place(bin_data, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx][0].append((x, y, z, w, h, d))
                    place_in_grid(bins[b_idx][1], x, y, z, w, h, d)
                    result[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                new_grid = make_grid()
                new_bin = [[(0, 0, 0, w, h, d)], new_grid]
                place_in_grid(new_grid, 0, 0, 0, w, h, d)
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    return result
# EVOLVE-BLOCK-END
