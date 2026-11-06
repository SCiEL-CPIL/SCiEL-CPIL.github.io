"""3D Bin Packing with vectorized candidate evaluation and local search."""
import time
import numpy as np
import random

def pack(boxes, container):
    """
    Place boxes into bins using FFD with vectorized collision checking.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0
    container_volume = W * H * D

    # Sort boxes by volume descending for FFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))

    # Each bin: list of (x, y, z, w, h, d) and a 3D occupancy grid
    bins = []
    grids = []

    def make_grid():
        return np.zeros((H, W, D), dtype=np.uint8)

    def can_place_grid(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[y:y+h, x:x+w, z:z+d].any():
            return False
        return True

    def place_grid(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 1

    def unplace_grid(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 0

    def generate_candidates(bin_placements, w, h, d):
        """Generate candidate (x, y, z) positions."""
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in bin_placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # Also add positions at the "floor" of each existing box
            candidates.add((px, py, pz))
        
        # Filter valid positions
        valid = []
        for (xc, yc, zc) in candidates:
            if xc + w <= W and yc + h <= H and zc + d <= D:
                valid.append((xc, yc, zc))
        return valid

    def find_best_position(bin_placements, grid, w, h, d):
        """Find the best position for a box in a bin."""
        if not bin_placements:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        candidates = generate_candidates(bin_placements, w, h, d)
        if not candidates:
            return None
        
        best_pos = None
        best_score = float('inf')
        
        for (xc, yc, zc) in candidates:
            if can_place_grid(grid, xc, yc, zc, w, h, d):
                # Score: prefer positions that minimize the "bounding box" extent
                # Use a weighted sum that prioritizes filling lower/inner positions
                score = xc * 1.0 + yc * 1.0 + zc * 1.0
                if score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    # Main packing loop - FFD
    result = [None] * n
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            grid = grids[b_idx]
            pos = find_best_position(bin_placements, grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize the resulting bounding box
                if bin_placements:
                    new_max_x = max(x + w, max(px + pw for (px, py, pz, pw, ph, pd) in bin_placements))
                    new_max_y = max(y + h, max(py + ph for (px, py, pz, pw, ph, pd) in bin_placements))
                    new_max_z = max(z + d, max(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements))
                else:
                    new_max_x, new_max_y, new_max_z = x + w, y + h, z + d
                fit_score = new_max_x + new_max_y + new_max_z
                
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            place_grid(grids[best_bin], x, y, z, w, h, d)
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            if w <= W and h <= H and d <= D:
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                place_grid(new_grid, 0, 0, 0, w, h, d)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in container - place at origin anyway (shouldn't happen)
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Handle any unplaced boxes (shouldn't happen if container is large enough)
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is None:
            for b_idx in range(len(bins)):
                pos = find_best_position(bins[b_idx], grids[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    place_grid(grids[b_idx], x, y, z, w, h, d)
                    result[idx] = (b_idx, x, y, z, w, h, d)
                    break
            else:
                new_grid = make_grid()
                bins.append([(0, 0, 0, w, h, d)])
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Local search: try to reduce number of bins with move and swap operations
    def compute_bin_volume(bin_placements):
        return sum(w * h * d for (_, _, _, w, h, d) in bin_placements)

    def try_move_box(source_bin, box_idx, target_bin):
        """Try to move a box from source_bin to target_bin."""
        x, y, z, w, h, d = bins[source_bin][box_idx]
        pos = find_best_position(bins[target_bin], grids[target_bin], w, h, d)
        if pos is not None:
            unplace_grid(grids[source_bin], x, y, z, w, h, d)
            bins[source_bin].pop(box_idx)
            nx, ny, nz = pos
            bins[target_bin].append((nx, ny, nz, w, h, d))
            place_grid(grids[target_bin], nx, ny, nz, w, h, d)
            # Update result
            for i in range(len(result)):
                bin_id, bx, by, bz, bw, bh, bd = result[i]
                if bin_id == source_bin and (bx, by, bz, bw, bh, bd) == (x, y, z, w, h, d):
                    result[i] = (target_bin, nx, ny, nz, w, h, d)
                    break
            return True
        return False

    def try_swap_boxes(bin1, idx1, bin2, idx2):
        """Try to swap two boxes between bins."""
        x1, y1, z1, w1, h1, d1 = bins[bin1][idx1]
        x2, y2, z2, w2, h2, d2 = bins[bin2][idx2]
        
        # Remove both boxes temporarily
        unplace_grid(grids[bin1], x1, y1, z1, w1, h1, d1)
        unplace_grid(grids[bin2], x2, y2, z2, w2, h2, d2)
        box1 = bins[bin1].pop(idx1)
        box2 = bins[bin2].pop(idx2)
        
        # Try to place box1 in bin2's space
        pos1_in_bin2 = find_best_position(bins[bin2], grids[bin2], w1, h1, d1)
        if pos1_in_bin2 is not None:
            # Try to place box2 in bin1's space
            pos2_in_bin1 = find_best_position(bins[bin1], grids[bin1], w2, h2, d2)
            if pos2_in_bin1 is not None:
                # Swap is valid
                nx1, ny1, nz1 = pos1_in_bin2
                nx2, ny2, nz2 = pos2_in_bin1
                bins[bin2].append((nx1, ny1, nz1, w1, h1, d1))
                place_grid(grids[bin2], nx1, ny1, nz1, w1, h1, d1)
                bins[bin1].append((nx2, ny2, nz2, w2, h2, d2))
                place_grid(grids[bin1], nx2, ny2, nz2, w2, h2, d2)
                
                # Update results
                for i in range(len(result)):
                    bin_id, bx, by, bz, bw, bh, bd = result[i]
                    if bin_id == bin1 and (bx, by, bz, bw, bh, bd) == (x1, y1, z1, w1, h1, d1):
                        result[i] = (bin2, nx1, ny1, nz1, w1, h1, d1)
                        break
                for i in range(len(result)):
                    bin_id, bx, by, bz, bw, bh, bd = result[i]
                    if bin_id == bin2 and (bx, by, bz, bw, bh, bd) == (x2, y2, z2, w2, h2, d2):
                        result[i] = (bin1, nx2, ny2, nz2, w2, h2, d2)
                        break
                return True
            else:
                # Rollback: put box2 back in bin2
                bins[bin2].append((x2, y2, z2, w2, h2, d2))
                place_grid(grids[bin2], x2, y2, z2, w2, h2, d2)
        else:
            # Rollback: put both boxes back
            bins[bin1].append((x1, y1, z1, w1, h1, d1))
            place_grid(grids[bin1], x1, y1, z1, w1, h1, d1)
            bins[bin2].append((x2, y2, z2, w2, h2, d2))
            place_grid(grids[bin2], x2, y2, z2, w2, h2, d2)
        
        return False

    if time.time() < deadline - 20:
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            
            # Phase 1: Move boxes to consolidate bins
            for b_idx in range(len(bins) - 1, -1, -1):
                if time.time() > deadline - 20:
                    break
                if not bins[b_idx]:
                    continue
                
                boxes_to_try = list(range(len(bins[b_idx])))
                random.shuffle(boxes_to_try)
                
                for box_idx in boxes_to_try:
                    if time.time() > deadline - 20:
                        break
                    if box_idx >= len(bins[b_idx]):
                        continue
                    
                    moved = False
                    for target_bin in range(b_idx):
                        if time.time() > deadline - 20:
                            break
                        if try_move_box(b_idx, box_idx, target_bin):
                            moved = True
                            changed = True
                            break
                    
                    if not bins[b_idx]:
                        bins.pop(b_idx)
                        grids.pop(b_idx)
                        for i in range(len(result)):
                            bin_id = result[i][0]
                            if bin_id > b_idx:
                                result[i] = (bin_id - 1,) + result[i][1:]
                        break

            # Phase 2: Try swaps between adjacent bins
            if time.time() < deadline - 20:
                for b_idx in range(len(bins) - 1):
                    if time.time() > deadline - 20:
                        break
                    if not bins[b_idx] or not bins[b_idx + 1]:
                        continue
                    
                    for idx1 in range(len(bins[b_idx])):
                        if time.time() > deadline - 20:
                            break
                        for idx2 in range(len(bins[b_idx + 1])):
                            if time.time() > deadline - 20:
                                break
                            if try_swap_boxes(b_idx, idx1, b_idx + 1, idx2):
                                changed = True
                                break
                        if changed:
                            break

    # Final local search: try to improve last bin's occupancy by moving boxes in
    if time.time() < deadline - 5 and len(bins) > 1:
        last_bin_idx = len(bins) - 1
        for b_idx in range(last_bin_idx):
            if time.time() > deadline - 5:
                break
            boxes_to_try = list(range(len(bins[b_idx])))
            random.shuffle(boxes_to_try)
            for box_idx in boxes_to_try:
                if time.time() > deadline - 5:
                    break
                if box_idx >= len(bins[b_idx]):
                    continue
                if try_move_box(b_idx, box_idx, last_bin_idx):
                    pass

    return result
