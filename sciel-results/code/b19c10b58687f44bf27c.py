import time
import numpy as np

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _find_position_grid(box, placements, grid, W, H, D):
    """
    Find the best position for a box in a bin using grid for overlap checks.
    Strategy: Bottom-Front-Left (BFL) - lowest z, then lowest y, then lowest x.
    Candidate coordinates are 0 and the end-faces of existing boxes.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate coordinates: 0 and right/top/back faces of existing boxes
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (_, px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    # Bottom-Front-Left: iterate z (lowest first), then y (lowest first), then x (lowest first)
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                # Check if the region is empty
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    # Safety margin: 145s limit for 150s total
    deadline = start_time + 145.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute volumes for sorting
    volumes = [_box_volume(b) for b in boxes]
    max_dims = [max(b) for b in boxes]
    
    def _pack_with_order(order, strategy_name):
        """
        Pack boxes in the given order.
        Returns (bins, bin_grids, result, score)
        """
        bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
        bin_grids = []  # List of numpy boolean arrays (W, H, D)
        result = [None] * n
        
        # Helper to update grid
        def _update_grid(grid, box, pos, add=True):
            x, y, z = pos
            w, h, d = box
            grid[x:x+w, y:y+h, z:z+d] = add
        
        for orig_idx in order:
            if time.time() > deadline - 10.0:
                # Emergency: place remaining boxes in new bins to ensure validity
                for oi in order:
                    if result[oi] is None:
                        bi = len(bins)
                        w, h, d = boxes[oi]
                        bins.append([(bi, 0, 0, 0, w, h, d)])
                        grid = np.zeros((W, H, D), dtype=bool)
                        _update_grid(grid, boxes[oi], (0, 0, 0))
                        bin_grids.append(grid)
                        result[oi] = (bi, 0, 0, 0, w, h, d)
                break
                
            box = boxes[orig_idx]
            w, h, d = box
            placed = False
            
            if bins:
                # Calculate remaining volume for each bin to sort by "tightness"
                # We want to try bins that are already full first (Best Fit Decreasing style)
                # to consolidate boxes into fewer bins.
                bin_remaining_vols = []
                for bi in range(len(bins)):
                    # Sum of occupied cells in grid
                    occ_vol = np.sum(bin_grids[bi])
                    rem_vol = W * H * D - occ_vol
                    bin_remaining_vols.append((rem_vol, bi))
                
                # Sort by remaining volume ascending (tightest bins first)
                bin_remaining_vols.sort(key=lambda x: x[0])
                
                for _, bi in bin_remaining_vols:
                    if time.time() > deadline - 5.0:
                        break
                    pos = _find_position_grid(box, bins[bi], bin_grids[bi], W, H, D)
                    if pos is not None:
                        x, y, z = pos
                        bins[bi].append((bi, x, y, z, w, h, d))
                        _update_grid(bin_grids[bi], box, pos)
                        result[orig_idx] = (bi, x, y, z, w, h, d)
                        placed = True
                        break
            
            if not placed:
                # Open new bin
                bi = len(bins)
                bins.append([(bi, 0, 0, 0, w, h, d)])
                grid = np.zeros((W, H, D), dtype=bool)
                _update_grid(grid, box, (0, 0, 0))
                bin_grids.append(grid)
                result[orig_idx] = (bi, 0, 0, 0, w, h, d)
        
        # Ensure all are placed (safety check)
        for i in range(n):
            if result[i] is None:
                bi = len(bins)
                w, h, d = boxes[i]
                bins.append([(bi, 0, 0, 0, w, h, d)])
                grid = np.zeros((W, H, D), dtype=bool)
                _update_grid(grid, boxes[i], (0, 0, 0))
                bin_grids.append(grid)
                result[i] = (bi, 0, 0, 0, w, h, d)
        
        # Calculate Score: B + 0.1 * (1 - last_bin_fill)
        occupied_bins = set(r[0] for r in result)
        B = len(occupied_bins)
        
        if B == 0:
            return bins, bin_grids, result, 0.0
        
        last_bin_id = max(r[0] for r in result)
        last_vol = np.sum(bin_grids[last_bin_id])
        container_vol = W * H * D
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        
        score = B + 0.1 * (1 - fill)
        
        return bins, bin_grids, result, score

    # Define different sorting strategies
    # 1. Volume Descending
    order1 = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    # 2. Max Dimension Descending, then Volume
    order2 = sorted(range(n), key=lambda i: (max_dims[i], volumes[i]), reverse=True)
    # 3. Volume Descending, then Max Dimension
    order3 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    # 4. Max Dimension Descending
    order4 = sorted(range(n), key=lambda i: max_dims[i], reverse=True)

    strategies = [
        (order1, "VolDesc"),
        (order2, "MaxDimVol"),
        (order3, "VolMaxDim"),
        (order4, "MaxDim"),
    ]

    best_score = float('inf')
    best_bins = None
    best_grids = None
    best_result = None

    # Run strategies
    for order, name in strategies:
        if time.time() > deadline - 20.0:
            break
            
        bins_try, grids_try, result_try, score_try = _pack_with_order(order, name)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_grids = grids_try
            best_result = result_try

    bins = best_bins
    bin_grids = best_grids
    result = best_result

    # Post-Optimization: Try to fill the last bin more
    # Score = B + 0.1 * (1 - fill_last)
    # We want to maximize fill_last.
    # We can move boxes from other bins to the last bin.
    # Note: Moving a box might not change B if the source bin becomes empty, but it usually doesn't hurt B if source bin still has boxes.
    # If source bin becomes empty, B decreases, which is even better!
    
    if time.time() < deadline - 5.0:
        last_bin_id = max(r[0] for r in result)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Get list of boxes not in last bin, sorted by volume descending (try to fill with big boxes first)
            candidates = [i for i in range(n) if result[i][0] != last_bin_id]
            candidates.sort(key=lambda i: volumes[i], reverse=True)
            
            for idx in candidates:
                if time.time() > deadline - 2.0:
                    break
                
                cur_bin_id = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                cur_pos = (result[idx][1], result[idx][2], result[idx][3])
                
                # 1. Remove from current bin
                _update_grid(bin_grids[cur_bin_id], box, cur_pos, add=False)
                # Remove from bins list
                target_placement = (cur_bin_id, cur_pos[0], cur_pos[1], cur_pos[2], w, h, d)
                bins[cur_bin_id] = [p for p in bins[cur_bin_id] if p != target_placement]
                
                # 2. Try to place in last bin
                pos_new = _find_position_grid(box, bins[last_bin_id], bin_grids[last_bin_id], W, H, D)
                
                if pos_new is not None:
                    # Move it
                    x, y, z = pos_new
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    _update_grid(bin_grids[last_bin_id], box, pos_new)
                    result[idx] = (last_bin_id, x, y, z, w, h, d)
                    improved = True
                    # We don't break here, we continue to try to move more boxes
                    # But we need to be careful: if we move too many, we might fill last bin completely.
                    # The loop condition handles this.
                else:
                    # Restore if it didn't fit
                    bins[cur_bin_id].append(target_placement)
                    _update_grid(bin_grids[cur_bin_id], box, cur_pos, add=True)
        
        # Check if any bins became empty and can be removed (relabeling handles this, but let's clean up)
        # The relabeling step will handle non-consecutive labels.

    # Relabel bins to be consecutive 0..B-1
    if result:
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
            
    return result
