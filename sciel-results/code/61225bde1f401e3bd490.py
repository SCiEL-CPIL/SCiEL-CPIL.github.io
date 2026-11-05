"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending (with tie-breaking by max dimension).
2. Use a per-bin 3D occupancy grid (NumPy boolean) for O(1) overlap checks.
3. Initial packing: Best-Fit-Decreasing with a "lowest z, then y, then x" placement
   heuristic that considers candidate positions derived from existing box boundaries.
4. Post-processing:
   - Attempt to consolidate bins by moving boxes from the highest-indexed bin into
     earlier bins to reduce B.
   - If consolidation isn't possible, try to fill the last bin more completely by
     moving boxes into it from other bins.
5. Relabel bins to be consecutive.
"""
import time
import numpy as np

def _find_position(box, placements, grid, W, H, D):
    """
    Find the best position for a box in a bin.
    Uses a bottom-left-front heuristic: lowest z, then lowest y, then lowest x.
    Candidate positions are derived from 0 and the faces of existing boxes.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Build candidate coordinate sets
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (_, px, py, pz, pw, ph, pd) in placements:
        if px + pw <= W:
            xs.add(px + pw)
        if py + ph <= H:
            ys.add(py + ph)
        if pz + pd <= D:
            zs.add(pz + pd)
    
    # Filter to valid positions
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    # Bottom-left-front: iterate z, then y, then x
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                # Check if the box fits without overlap
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, tie-break by max dimension descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], max(t[1])), reverse=True)
    
    bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
    bin_grids = []  # List of numpy boolean arrays
    result = [None] * n
    bin_volumes = []  # Track volume of each bin
    
    def _update_grid(grid, box, pos, add=True):
        x, y, z = pos
        w, h, d = box
        grid[x:x+w, y:y+h, z:z+d] = add
    
    # Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency: place remaining boxes each in new bins
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    grid = np.zeros((W, H, D), dtype=bool)
                    _update_grid(grid, ob, (0, 0, 0))
                    bin_grids.append(grid)
                    bin_volumes.append(w*h*d)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
        
        w, h, d = box
        placed = False
        
        if bins:
            # Try to place in existing bins, sorted by remaining volume (tightest first)
            bin_indices = sorted(range(len(bins)), key=lambda bi: W*H*D - bin_volumes[bi])
            
            for bi in bin_indices:
                if time.time() > deadline - 3.0:
                    break
                pos = _find_position(box, bins[bi], bin_grids[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    _update_grid(bin_grids[bi], box, pos)
                    bin_volumes[bi] += w*h*d
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
            bin_volumes.append(w*h*d)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0, 0, 0))
            bin_grids.append(grid)
            bin_volumes.append(w*h*d)
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Consolidate bins - try to empty the highest-indexed bins
    if time.time() < deadline - 5.0:
        # Try to empty bins from highest index down to 1 (keep bin 0)
        for src_bin in range(len(bins)-1, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if not bins[src_bin]:
                continue
            
            boxes_to_move = list(bins[src_bin])
            if not boxes_to_move:
                continue
            
            # Sort boxes to move by volume descending
            boxes_to_move.sort(key=lambda p: p[4]*p[5]*p[6], reverse=True)
            
            can_move_all = True
            saved_states = []
            
            for p in boxes_to_move:
                bi, x, y, z, w, h, d = p
                box = (w, h, d)
                
                # Find which box index this is
                box_idx = None
                for i in range(n):
                    if result[i] and result[i][0] == src_bin and result[i][1] == x \
                       and result[i][2] == y and result[i][3] == z \
                       and result[i][4] == w and result[i][5] == h and result[i][6] == d:
                        box_idx = i
                        break
                if box_idx is None:
                    can_move_all = False
                    break
                
                # Remove from source
                _update_grid(bin_grids[src_bin], box, (x, y, z), add=False)
                bin_volumes[src_bin] -= w*h*d
                saved_states.append((box_idx, x, y, z, w, h, d))
                
                # Try to place in other bins (not src_bin)
                placed = False
                for dst_bin in range(len(bins)):
                    if dst_bin == src_bin:
                        continue
                    pos = _find_position(box, bins[dst_bin], bin_grids[dst_bin], W, H, D)
                    if pos is not None:
                        nx, ny, nz = pos
                        bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                        _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                        bin_volumes[dst_bin] += w*h*d
                        result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                        placed = True
                        break
                
                if not placed:
                    # Restore all moved boxes
                    for bi2, xi, yi, zi, wi, hi, di in saved_states:
                        _update_grid(bin_grids[src_bin], (wi, hi, di), (xi, yi, zi), add=True)
                        bin_volumes[src_bin] += wi*hi*di
                    can_move_all = False
                    break
            
            if can_move_all:
                bins[src_bin] = []
                bin_volumes[src_bin] = 0
    
    # Phase 3: Try to improve last bin fill by moving boxes into it
    if time.time() < deadline - 3.0 and len(bins) > 1:
        # Identify the last occupied bin
        last_bin_id = None
        for bi in range(len(bins)-1, -1, -1):
            if bins[bi]:
                last_bin_id = bi
                break
        
        if last_bin_id is not None and last_bin_id > 0:
            # Try to move boxes from other bins into the last bin
            moved_any = True
            while moved_any and time.time() < deadline - 2.0:
                moved_any = False
                # Sort candidates by box volume descending
                candidates = sorted(
                    [i for i in range(n) if result[i][0] != last_bin_id],
                    key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2],
                    reverse=True
                )
                for idx in candidates:
                    if time.time() > deadline - 2.0:
                        break
                    
                    cur_bin = result[idx][0]
                    box = boxes[idx]
                    w, h, d = box
                    box_pos = result[idx][1:4]
                    
                    # Temporarily remove from source bin
                    _update_grid(bin_grids[cur_bin], box, box_pos, add=False)
                    bin_volumes[cur_bin] -= w*h*d
                    
                    # Try to place in last bin
                    pos = _find_position(box, bins[last_bin_id], bin_grids[last_bin_id], W, H, D)
                    
                    if pos is not None:
                        # Move it
                        x, y, z = pos
                        bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                        _update_grid(bin_grids[last_bin_id], box, pos)
                        bin_volumes[last_bin_id] += w*h*d
                        # Remove from source bin's placement list
                        bins[cur_bin] = [p for p in bins[cur_bin] 
                                         if not (p[1] == box_pos[0] and p[2] == box_pos[1] 
                                                 and p[3] == box_pos[2] and p[4] == w 
                                                 and p[5] == h and p[6] == d)]
                        result[idx] = (last_bin_id, x, y, z, w, h, d)
                        moved_any = True
                        break
                    else:
                        # Restore if it didn't fit
                        _update_grid(bin_grids[cur_bin], box, box_pos, add=True)
                        bin_volumes[cur_bin] += w*h*d
    
    # Phase 4: Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
    
    return result
