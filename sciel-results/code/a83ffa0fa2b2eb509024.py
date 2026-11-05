"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending.
2. Place boxes into bins using a Best-Fit-Decreasing heuristic with grid-based overlap checks.
3. Post-processing:
   - Move boxes from other bins into the last bin to increase its fill ratio.
   - Try to consolidate bins to reduce B.
4. Relabel bins to be consecutive.
"""
import time
import numpy as np
from scipy import ndimage

def _can_place_grid(box, pos, grid, W, H, D):
    """Check if box with dims (w,h,d) can be placed at (x,y,z) using grid."""
    x, y, z = pos
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False
    if bool(grid[x:x+w, y:y+h, z:z+d].any()):
        return False
    return True

def _find_position_grid(box, placements, grid, W, H, D):
    """
    Find the best position for a box in a bin using grid for overlap checks.
    Strategy: Bottom-Fill - lowest z, then lowest y, then lowest x.
    Generate candidate coordinates from existing box boundaries.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Candidate coordinates: 0 and right/top/front faces of existing boxes
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
    
    # Bottom-Fill: iterate z (lowest first), then y (lowest first), then x (lowest first)
    # Return the first valid position found
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if not grid[x:x+w, y:y+h, z:z+d].any():
                    return (x, y, z)
    
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
    bin_grids = []  # List of numpy boolean arrays (W, H, D)
    result = [None] * n
    
    def _update_grid(grid, box, pos, add=True):
        x, y, z = pos
        w, h, d = box
        grid[x:x+w, y:y+h, z:z+d] = add
    
    # 2. Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency placement: each in new bin
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    grid = np.zeros((W, H, D), dtype=bool)
                    _update_grid(grid, ob, (0, 0, 0))
                    bin_grids.append(grid)
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Estimate remaining volume for each bin using grid sum
            bin_vols = []
            for bi in range(len(bins)):
                vol = np.sum(bin_grids[bi])
                bin_vols.append((W*H*D - vol, bi))
            
            # Sort by remaining volume ascending (tightest bins first)
            bin_vols.sort(key=lambda x: x[0])
            
            for _, bi in bin_vols:
                if time.time() > deadline - 3.0:
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
            
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            _update_grid(grid, boxes[i], (0, 0, 0))
            bin_grids.append(grid)
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    # 3. Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 5.0:
        # Identify last bin (highest label)
        last_bin_id = max(r[0] for r in result)
        
        # Try to move boxes from other bins into the last bin to increase its fill ratio
        moved_any = True
        while moved_any and time.time() < deadline - 2.0:
            moved_any = False
            # Sort candidates by box volume descending to fill with larger boxes first
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
                
                # Try to place in last bin
                pos = _find_position_grid(box, bins[last_bin_id], bin_grids[last_bin_id], W, H, D)
                
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    _update_grid(bin_grids[last_bin_id], box, pos)
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
        
        # 3b. Try to consolidate bins: eliminate non-last bins by moving their boxes elsewhere
        if time.time() < deadline - 3.0:
            # Check if any bin is now empty
            for bi in range(len(bins)):
                if not bins[bi]:
                    continue
            
            # Try to empty non-last bins (highest index first, excluding last_bin_id)
            non_last_bins = sorted([bi for bi in range(len(bins)) if bi != last_bin_id and bins[bi]], reverse=True)
            
            for src_bin in non_last_bins:
                if time.time() > deadline - 3.0:
                    break
                
                boxes_to_move = list(bins[src_bin])
                if not boxes_to_move:
                    continue
                    
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
                    saved_states.append((box_idx, x, y, z, w, h, d))
                    
                    # Try to place in other bins (not src_bin)
                    placed = False
                    for dst_bin in range(len(bins)):
                        if dst_bin == src_bin:
                            continue
                        pos = _find_position_grid(box, bins[dst_bin], bin_grids[dst_bin], W, H, D)
                        if pos is not None:
                            nx, ny, nz = pos
                            bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                            _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                            result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                            placed = True
                            break
                    
                    if not placed:
                        # Restore all moved boxes
                        for bi2, xi, yi, zi, wi, hi, di in saved_states:
                            _update_grid(bin_grids[src_bin], (wi, hi, di), (xi, yi, zi), add=True)
                        can_move_all = False
                        break
                
                if can_move_all:
                    bins[src_bin] = []
    
    # 4. Relabel bins to be consecutive
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
