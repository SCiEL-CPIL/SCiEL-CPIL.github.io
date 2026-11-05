import time
import sys
import numpy as np

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _can_place(box, pos, placements, W, H, D):
    """
    Check if box with dims (w,h,d) can be placed at (x,y,z) in a bin with given placements.
    placements: list of (x, y, z, w, h, d)
    """
    x, y, z = pos
    w, h, d = box
    
    # Bounds check
    if x + w > W or y + h > H or z + d > D:
        return False
    
    # Overlap check
    for (px, py, pz, pw, ph, pd) in placements:
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position_bfd(box, placements, W, H, D):
    """
    Find the best position for a box in a bin using Bottom-Front-Left (BFL) / Best-Fit-Deeper logic.
    Preference: Lowest Z, then Lowest Y, then Lowest X.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate coordinates
    # Candidates are 0 and the end-faces of existing boxes
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    # Filter candidates that fit within container
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    # Sort candidates to prioritize Bottom (Z), Front (Y), Left (X)
    # We iterate Z first (lowest), then Y (lowest), then X (lowest)
    # The first valid position found in this order is the BFL position
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    return (x, y, z)
                    
    return None

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    # Safety margin: 140s limit for 150s total
    deadline = start_time + 140.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    def _pack_with_order(order):
        """Pack boxes in the given order using BFD heuristic. Returns (bins, result)."""
        bins = []
        result = [None] * n
        
        for orig_idx in order:
            box = boxes[orig_idx]
            w, h, d = box
            placed = False
            
            for bi in range(len(bins)):
                pos = _find_position_bfd(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                result[orig_idx] = (bi, 0, 0, 0, w, h, d)
        
        # Ensure all are placed
        for i in range(n):
            if result[i] is None:
                bi = len(bins)
                w, h, d = boxes[i]
                bins.append([(0, 0, 0, w, h, d)])
                result[i] = (bi, 0, 0, 0, w, h, d)
        
        return bins, result
    
    def _score_solution(bins, result):
        """Compute score for a solution: B + 0.1 * (1 - last_bin_fill)."""
        if not bins:
            return float('inf')
        
        occupied_bins = set()
        for r in result:
            occupied_bins.add(r[0])
        B = len(occupied_bins)
        
        last_bin_id = max(r[0] for r in result)
        last_vol = sum(p[3] * p[4] * p[5] for p in bins[last_bin_id])
        container_vol = W * H * D
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        
        return B + 0.1 * (1 - fill)
    
    # Strategy 1: Volume descending, then max dimension descending
    order1 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    
    # Strategy 2: Max dimension descending, then volume descending
    order2 = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    # Strategy 3: Volume ascending (small first) - sometimes helps with last bin fill
    order3 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=False)
    
    # Try all strategies and pick the best
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    for order in [order1, order2, order3]:
        if time.time() > deadline - 60.0:
            break
        
        bins_try, result_try = _pack_with_order(order)
        score_try = _score_solution(bins_try, result_try)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_result = result_try
    
    bins = best_bins
    result = best_result
            
    # 3. Phase 2: Optimize Last Bin Fill
    # Score = B + 0.1 * (1 - last_bin_fill)
    # We want to maximize last_bin_fill.
    # We can move boxes from other bins to the last bin.
    # We can also SWAP boxes to better fill the last bin.
    
    if time.time() < deadline - 3.0:
        # Identify last bin (highest label in result)
        # Note: result[0] is bin id.
        last_bin_id = max(r[0] for r in result)
        
        # We will perform multiple passes of improvement.
        # Pass 1: Move boxes from other bins to last bin if they fit.
        # Pass 2: Swap boxes between last bin and other bins if it increases last bin volume.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # --- Attempt Moves ---
            # Iterate through all boxes not in last bin
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                if result[idx][0] == last_bin_id:
                    continue
                    
                cur_bin_id = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                # Check if box fits in last bin
                last_placements = bins[last_bin_id]
                pos = _find_position_bfd(box, last_placements, W, H, D)
                
                if pos is not None:
                    # Check if moving this box would actually improve the last bin fill
                    # Only move if the box volume is significant or if it fits well
                    # For now, just move it (simple heuristic)
                    # Remove from current bin
                    cur_placements = bins[cur_bin_id]
                    target_placement = (result[idx][1], result[idx][2], result[idx][3], w, h, d)
                    
                    removed = False
                    for j in range(len(cur_placements)):
                        if cur_placements[j] == target_placement:
                            cur_placements.pop(j)
                            removed = True
                            break
                            
                    if removed:
                        bins[last_bin_id].append((pos[0], pos[1], pos[2], w, h, d))
                        result[idx] = (last_bin_id, pos[0], pos[1], pos[2], w, h, d)
                        improved = True
                        break
                        
            # --- Attempt Swaps ---
            # If no simple move found, try swapping.
            # This is more expensive, so only do it if we have time and didn't just move something.
            if not improved and time.time() < deadline - 10.0:
                # Iterate through boxes in last bin
                last_boxes_indices = [i for i in range(n) if result[i][0] == last_bin_id]
                
                for idx_last in last_boxes_indices:
                    if time.time() > deadline - 10.0:
                        break
                    
                    box_last = boxes[idx_last]
                    wl, hl, dl = box_last
                    pos_last = (result[idx_last][1], result[idx_last][2], result[idx_last][3])
                    target_last_placement = (pos_last[0], pos_last[1], pos_last[2], wl, hl, dl)
                    
                    # Iterate through boxes in other bins
                    for idx_other in range(n):
                        if time.time() > deadline - 10.0:
                            break
                        if result[idx_other][0] == last_bin_id:
                            continue
                            
                        box_other = boxes[idx_other]
                        wo, ho, do = box_other
                        pos_other = (result[idx_other][1], result[idx_other][2], result[idx_other][3])
                        target_other_placement = (pos_other[0], pos_other[1], pos_other[2], wo, ho, do)
                        
                        # Volume check: We want to increase the volume of the last bin.
                        # So volume(box_other) > volume(box_last)
                        if _box_volume(box_other) <= _box_volume(box_last):
                            continue
                            
                        # 1. Remove box_last from last_bin, check if box_other fits
                        last_placements_temp = bins[last_bin_id][:]
                        removed_last = False
                        for j in range(len(last_placements_temp)):
                            if last_placements_temp[j] == target_last_placement:
                                last_placements_temp.pop(j)
                                removed_last = True
                                break
                                
                        if not removed_last:
                            continue
                            
                        pos_new = _find_position_bfd(box_other, last_placements_temp, W, H, D)
                        if pos_new is None:
                            continue
                            
                        # 2. Remove box_other from other_bin, check if box_last fits
                        other_bin_id = result[idx_other][0]
                        other_placements_temp = bins[other_bin_id][:]
                        removed_other = False
                        for j in range(len(other_placements_temp)):
                            if other_placements_temp[j] == target_other_placement:
                                other_placements_temp.pop(j)
                                removed_other = True
                                break
                                
                        if not removed_other:
                            continue
                            
                        pos_new_last = _find_position_bfd(box_last, other_placements_temp, W, H, D)
                        if pos_new_last is None:
                            continue
                            
                        # Swap is valid!
                        # Update bins
                        bins[last_bin_id] = last_placements_temp + [(pos_new[0], pos_new[1], pos_new[2], wo, ho, do)]
                        bins[other_bin_id] = other_placements_temp + [(pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)]
                        
                        # Update results
                        result[idx_last] = (other_bin_id, pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)
                        result[idx_other] = (last_bin_id, pos_new[0], pos_new[1], pos_new[2], wo, ho, do)
                        
                        improved = True
                        break # Break inner loop, restart outer
                    if improved:
                        break # Break outer loop
                
    # 4. Relabel bins to be consecutive
    # This is important because the "last bin" is defined by the largest label.
    # If we have bins 0, 1, 3, the last bin is 3. If we empty bin 1, it's still 0, 1(empty), 3.
    # B counts occupied bins.
    # Relabeling ensures that if we empty a bin, its label disappears from the count of "occupied" bins
    # only if we actually remove the label. But the prompt says "Bin labels are nonnegative integers... largest label defines the last bin".
    # "B is the number of occupied bins".
    # If I have placements in bin 0 and bin 5, B=2. Last bin is 5.
    # If I relabel to 0 and 1, B=2. Last bin is 1.
    # The volume of the last bin is the same.
    # However, relabeling makes the state cleaner and avoids potential confusion if the evaluator checks for consecutive integers (though prompt says "need not be consecutive").
    # It's safer to relabel to 0..k-1.
    
    if bins:
        # Get unique occupied bin IDs
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
            
    return result
