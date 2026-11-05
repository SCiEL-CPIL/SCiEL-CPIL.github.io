"""
3D Bin Packing Solution
Strategy:
1. Sort boxes by volume descending.
2. Use a Best-Fit-Decreasing heuristic with efficient candidate position generation.
3. For each box, find the position that minimizes (z, y, x) - pack towards origin.
4. Post-processing: try to consolidate bins and maximize last-bin fill.
"""
import time
import numpy as np
import sys

def _find_position(box, placements, W, H, D):
    """
    Find the best position for a box in a bin.
    Strategy: Generate candidate positions from existing box boundaries,
    then pick the lexicographically smallest (z, y, x) that fits.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    # Generate candidate coordinates from box boundaries
    xs = set([0])
    ys = set([0])
    zs = set([0])

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

    # For each candidate position, check if it overlaps with any existing box
    # Sort by (z, y, x) to find the lowest position first
    candidates = []
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                candidates.append((z, y, x))

    candidates.sort()

    for z, y, x in candidates:
        # Check overlap with all placements
        fits = True
        for (_, px, py, pz, pw, ph, pd) in placements:
            if x < px + pw and px < x + w and \
               y < py + ph and py < y + h and \
               z < pz + pd and pz < z + d:
                fits = False
                break
        if fits:
            return (x, y, z)

    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Sort boxes by volume descending
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    bins = []  # List of lists of placements: (bin_id, x, y, z, w, h, d)
    result = [None] * n
    
    # 2. Phase 1: Initial Packing (Best Fit Decreasing)
    for orig_idx, box in indexed_boxes:
        if time.time() > deadline - 5.0:
            # Emergency placement: each in new bin
            for oi, ob in indexed_boxes:
                if result[oi] is None:
                    bi = len(bins)
                    w, h, d = ob
                    bins.append([(bi, 0, 0, 0, w, h, d)])
                    result[oi] = (bi, 0, 0, 0, w, h, d)
            break
            
        w, h, d = box
        placed = False
        
        if bins:
            # Try existing bins, sorted by current occupied volume (descending)
            # This is "best fit" - prefer the fullest bin that can accommodate the box
            bin_indices = sorted(range(len(bins)), key=lambda bi: -sum(
                p[4]*p[5]*p[6] for p in bins[bi]
            ))
            
            for bi in bin_indices:
                if time.time() > deadline - 3.0:
                    break
                pos = _find_position(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
            
    # Ensure all are placed (safety)
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[i] = (bi, 0, 0, 0, w, h, d)
            
    # 3. Phase 2: Try to consolidate bins
    if time.time() < deadline - 10.0:
        # Try to move boxes from higher-indexed bins to lower-indexed bins
        # to reduce the total number of bins
        for src_bin in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if not bins[src_bin]:
                continue
            
            # Try to move each box from src_bin to an earlier bin
            boxes_to_try = list(bins[src_bin])
            for p in boxes_to_try:
                if time.time() > deadline - 5.0:
                    break
                
                bi, x, y, z, w, h, d = p
                box = (w, h, d)
                
                # Find which box index this is
                box_idx = None
                for i in range(n):
                    if result[i] is not None and result[i][0] == bi and \
                       result[i][1] == x and result[i][2] == y and result[i][3] == z and \
                       result[i][4] == w and result[i][5] == h and result[i][6] == d:
                        box_idx = i
                        break
                
                if box_idx is None:
                    continue
                
                # Remove from source bin
                bins[src_bin].remove(p)
                
                # Try to place in earlier bins
                moved = False
                for dst_bin in range(src_bin):
                    if time.time() > deadline - 5.0:
                        break
                    pos = _find_position(box, bins[dst_bin], W, H, D)
                    if pos is not None:
                        nx, ny, nz = pos
                        bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                        result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                        moved = True
                        break
                
                if not moved:
                    # Restore
                    bins[src_bin].append(p)
                    result[box_idx] = (bi, x, y, z, w, h, d)
            
            # If bin is now empty, it will be removed in relabeling
            
    # 4. Phase 3: Optimize last bin fill
    if time.time() < deadline - 5.0:
        # Identify last bin (highest label)
        last_bin_id = max(r[0] for r in result if r is not None)
        
        moved_any = True
        while moved_any and time.time() < deadline - 2.0:
            moved_any = False
            candidates = [i for i in range(n) if result[i] is not None and result[i][0] != last_bin_id]
            for idx in candidates:
                if time.time() > deadline - 2.0:
                    break
                
                cur_bin = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                # Remove from source bin
                src_placement = result[idx]
                bins[cur_bin].remove(src_placement)
                
                # Try to place in last bin
                pos = _find_position(box, bins[last_bin_id], W, H, D)
                
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[last_bin_id].append((last_bin_id, x, y, z, w, h, d))
                    result[idx] = (last_bin_id, x, y, z, w, h, d)
                    moved_any = True
                    break
                else:
                    # Restore
                    bins[cur_bin].append(src_placement)
                    result[idx] = src_placement
        
    # 5. Relabel bins to be consecutive
    occupied_bins = sorted(set(r[0] for r in result if r is not None))
    label_map = {old: new for new, old in enumerate(occupied_bins)}
    
    final_result = []
    for i in range(n):
        if result[i] is not None:
            old_bin = result[i][0]
            new_bin = label_map.get(old_bin, 0)
            final_result.append((new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6]))
        else:
            # Should not happen, but safety fallback
            w, h, d = boxes[i]
            final_result.append((0, 0, 0, 0, w, h, d))
    
    return final_result
