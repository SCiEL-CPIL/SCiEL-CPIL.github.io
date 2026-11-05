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

def _find_position_bfl(box, placements, W, H, D):
    """
    Find the best position for a box in a bin using Bottom-Front-Left (BFL) logic.
    Preference: Lowest Z, then Lowest Y, then Lowest X.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate coordinates
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    return (x, y, z)
                    
    return None

def _find_position_max_contact(box, placements, W, H, D):
    """
    Find the position that maximizes contact with existing boxes and walls.
    This tends to pack tighter than BFL in some cases.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    if not placements:
        return (0, 0, 0)
        
    best_pos = None
    best_score = float('inf')
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    # Score: maximize contact (minimize gap to nearest surfaces)
                    min_gap = float('inf')
                    for (px, py, pz, pw, ph, pd) in placements:
                        # Gap in x
                        if x >= px + pw:
                            gap_x = x - (px + pw)
                        elif x + w <= px:
                            gap_x = px - (x + w)
                        else:
                            gap_x = 0
                        # Gap in y
                        if y >= py + ph:
                            gap_y = y - (py + ph)
                        elif y + h <= py:
                            gap_y = py - (y + h)
                        else:
                            gap_y = 0
                        # Gap in z
                        if z >= pz + pd:
                            gap_z = z - (pz + pd)
                        elif z + d <= pz:
                            gap_z = pz - (z + d)
                        else:
                            gap_z = 0
                        max_gap = max(gap_x, gap_y, gap_z)
                        if max_gap < min_gap:
                            min_gap = max_gap
                            if min_gap == 0:
                                break
                    
                    wall_gap = min(x, y, z, W - (x + w), H - (y + h), D - (z + d))
                    total_score = max(min_gap, wall_gap)
                    
                    # Tie-break with BFL
                    score = total_score * 1000000 + z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    def _pack_with_order(order, finder_func):
        """Pack boxes in the given order using specified position finder. Returns (bins, result)."""
        bins = []
        result = [None] * n
        
        for orig_idx in order:
            box = boxes[orig_idx]
            w, h, d = box
            placed = False
            
            for bi in range(len(bins)):
                pos = finder_func(box, bins[bi], W, H, D)
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
        
        return bins, result
    
    def _score_solution(result):
        """Compute score for a solution: B + 0.1 * (1 - last_bin_fill)."""
        if not result:
            return float('inf')
        
        occupied_bins = set(r[0] for r in result)
        B = len(occupied_bins)
        
        last_bin_id = max(r[0] for r in result)
        last_vol = sum(r[4] * r[5] * r[6] for r in result if r[0] == last_bin_id)
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        
        return B + 0.1 * (1 - fill)
    
    def _remove_from_bin(bins, bin_id, target):
        """Remove a specific placement from a bin. Returns True if removed."""
        pl = bins[bin_id]
        for j in range(len(pl)):
            if pl[j] == target:
                pl.pop(j)
                return True
        return False
    
    def _try_move_to_last_bin(result, bins, last_bin_id, idx):
        """Try to move box idx to the last bin. Returns True if moved."""
        if result[idx][0] == last_bin_id:
            return False
        
        box = boxes[idx]
        w, h, d = box
        cur_bin_id = result[idx][0]
        
        pos = _find_position_max_contact(box, bins[last_bin_id], W, H, D)
        if pos is None:
            pos = _find_position_bfl(box, bins[last_bin_id], W, H, D)
        if pos is None:
            return False
        
        target = (result[idx][1], result[idx][2], result[idx][3], w, h, d)
        if not _remove_from_bin(bins, cur_bin_id, target):
            return False
        
        bins[last_bin_id].append((pos[0], pos[1], pos[2], w, h, d))
        result[idx] = (last_bin_id, pos[0], pos[1], pos[2], w, h, d)
        return True
    
    def _try_swap(result, bins, last_bin_id, idx_last, idx_other):
        """Try to swap box idx_last (in last bin) with box idx_other (in other bin).
        Only if it increases last bin volume. Returns True if swapped."""
        box_last = boxes[idx_last]
        box_other = boxes[idx_other]
        
        if _box_volume(box_other) <= _box_volume(box_last):
            return False
        
        wl, hl, dl = box_last
        wo, ho, do = box_other
        
        pos_last = (result[idx_last][1], result[idx_last][2], result[idx_last][3])
        pos_other = (result[idx_other][1], result[idx_other][2], result[idx_other][3])
        
        target_last = (pos_last[0], pos_last[1], pos_last[2], wl, hl, dl)
        target_other = (pos_other[0], pos_other[1], pos_other[2], wo, ho, do)
        
        other_bin_id = result[idx_other][0]
        
        # Simulate: remove box_last from last bin, try to place box_other
        last_temp = list(bins[last_bin_id])
        removed_last = False
        for j in range(len(last_temp)):
            if last_temp[j] == target_last:
                last_temp.pop(j)
                removed_last = True
                break
        if not removed_last:
            return False
        
        pos_new = _find_position_max_contact(box_other, last_temp, W, H, D)
        if pos_new is None:
            pos_new = _find_position_bfl(box_other, last_temp, W, H, D)
        if pos_new is None:
            return False
        
        # Simulate: remove box_other from other bin, try to place box_last
        other_temp = list(bins[other_bin_id])
        removed_other = False
        for j in range(len(other_temp)):
            if other_temp[j] == target_other:
                other_temp.pop(j)
                removed_other = True
                break
        if not removed_other:
            return False
        
        pos_new_last = _find_position_max_contact(box_last, other_temp, W, H, D)
        if pos_new_last is None:
            pos_new_last = _find_position_bfl(box_last, other_temp, W, H, D)
        if pos_new_last is None:
            return False
        
        # Apply swap
        bins[last_bin_id] = last_temp + [(pos_new[0], pos_new[1], pos_new[2], wo, ho, do)]
        bins[other_bin_id] = other_temp + [(pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)]
        result[idx_last] = (other_bin_id, pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)
        result[idx_other] = (last_bin_id, pos_new[0], pos_new[1], pos_new[2], wo, ho, do)
        return True
    
    def _compaction_pass(result, bins):
        """Try to reduce bin count by moving all boxes from one bin to others.
        Returns (bins, result, success)."""
        bin_vols = {}
        for r in result:
            bid = r[0]
            vol = r[4] * r[5] * r[6]
            bin_vols[bid] = bin_vols.get(bid, 0) + vol
        
        if len(bin_vols) <= 1:
            return bins, result, False
        
        target_bin = min(bin_vols, key=bin_vols.get)
        target_indices = [i for i in range(n) if result[i][0] == target_bin]
        
        new_bins = [list(pl) for pl in bins]
        new_result = list(result)
        
        all_moved = True
        for idx in target_indices:
            box = boxes[idx]
            w, h, d = box
            moved = False
            for bi in range(len(new_bins)):
                if bi == target_bin:
                    continue
                pos = _find_position_max_contact(box, new_bins[bi], W, H, D)
                if pos is None:
                    pos = _find_position_bfl(box, new_bins[bi], W, H, D)
                if pos is not None:
                    new_bins[bi].append((pos[0], pos[1], pos[2], w, h, d))
                    new_result[idx] = (bi, pos[0], pos[1], pos[2], w, h, d)
                    moved = True
                    break
            if not moved:
                all_moved = False
                break
        
        if all_moved:
            return new_bins, new_result, True
        return bins, result, False
    
    # Phase 1: Try multiple ordering strategies with both position finders
    order1 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    order2 = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    order3 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=False)
    order4 = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), _box_volume(boxes[i])), reverse=True)
    
    strategies = [
        (order1, _find_position_bfl),
        (order1, _find_position_max_contact),
        (order2, _find_position_bfl),
        (order2, _find_position_max_contact),
        (order3, _find_position_bfl),
        (order4, _find_position_max_contact),
    ]
    
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    initial_deadline = start_time + 55.0
    
    for order, finder in strategies:
        if time.time() > initial_deadline:
            break
        
        bins_try, result_try = _pack_with_order(order, finder)
        score_try = _score_solution(result_try)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_result = result_try
    
    bins = best_bins
    result = best_result
    
    # Phase 2: Local search to optimize
    if time.time() < deadline - 5.0:
        last_bin_id = max(r[0] for r in result)
        
        # Pass A: Move boxes to last bin to increase its fill
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 10.0:
                    break
                if _try_move_to_last_bin(result, bins, last_bin_id, idx):
                    improved = True
                    last_bin_id = max(r[0] for r in result)
                    break
        
        # Pass B: Swap boxes to increase last bin volume
        if time.time() < deadline - 10.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                last_bin_id = max(r[0] for r in result)
                last_indices = [i for i in range(n) if result[i][0] == last_bin_id]
                
                for idx_last in last_indices:
                    if time.time() > deadline - 5.0:
                        break
                    for idx_other in range(n):
                        if time.time() > deadline - 5.0:
                            break
                        if result[idx_other][0] == last_bin_id:
                            continue
                        if _try_swap(result, bins, last_bin_id, idx_last, idx_other):
                            improved = True
                            break
                    if improved:
                        break
        
        # Pass C: Compaction - try to eliminate the least-filled bin
        if time.time() < deadline - 3.0:
            new_bins, new_result, success = _compaction_pass(result, bins)
            if success:
                new_score = _score_solution(new_result)
                cur_score = _score_solution(result)
                if new_score < cur_score:
                    bins = new_bins
                    result = new_result
        
        # Pass D: Second compaction attempt
        if time.time() < deadline - 2.0:
            new_bins, new_result, success = _compaction_pass(result, bins)
            if success:
                new_score = _score_solution(new_result)
                cur_score = _score_solution(result)
                if new_score < cur_score:
                    bins = new_bins
                    result = new_result
    
    # Relabel bins to be consecutive 0..k-1
    if bins and result:
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3],
                             result[i][4], result[i][5], result[i][6])
    
    return result
