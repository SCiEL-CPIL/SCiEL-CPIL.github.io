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
    Find the best position for a box in a bin using a weighted scoring heuristic.
    Preference: Lowest Z (weight 3), then Lowest Y (weight 2), then Lowest X (weight 1).
    Uses normalized coordinates to make the scoring dimension-agnostic.
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
    
    # Score each valid position: lower is better
    # Normalize by container dimensions so weights are comparable
    best_score = float('inf')
    best_pos = None
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    # Weighted score: z matters most, then y, then x
                    score = (z / max(D, 1)) * 3.0 + (y / max(H, 1)) * 2.0 + (x / max(W, 1)) * 1.0
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
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
        
        pos = _find_position_bfd(box, bins[last_bin_id], W, H, D)
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
        last_temp = bins[last_bin_id][:]
        removed_last = False
        for j in range(len(last_temp)):
            if last_temp[j] == target_last:
                last_temp.pop(j)
                removed_last = True
                break
        if not removed_last:
            return False
        
        pos_new = _find_position_bfd(box_other, last_temp, W, H, D)
        if pos_new is None:
            return False
        
        # Simulate: remove box_other from other bin, try to place box_last
        other_temp = bins[other_bin_id][:]
        removed_other = False
        for j in range(len(other_temp)):
            if other_temp[j] == target_other:
                other_temp.pop(j)
                removed_other = True
                break
        if not removed_other:
            return False
        
        pos_new_last = _find_position_bfd(box_last, other_temp, W, H, D)
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
        # Find the least-filled bin (by volume) among occupied bins
        bin_vols = {}
        for r in result:
            bid = r[0]
            vol = r[4] * r[5] * r[6]
            bin_vols[bid] = bin_vols.get(bid, 0) + vol
        
        if len(bin_vols) <= 1:
            return bins, result, False
        
        # Target: the bin with smallest volume
        target_bin = min(bin_vols, key=bin_vols.get)
        
        # Get boxes in target bin
        target_indices = [i for i in range(n) if result[i][0] == target_bin]
        
        # Try to move each box to another bin
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
                pos = _find_position_bfd(box, new_bins[bi], W, H, D)
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
    
    # Phase 1: Try multiple ordering strategies
    # Strategy 1: Volume descending, then max dimension descending
    order1 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    
    # Strategy 2: Max dimension descending, then volume descending
    order2 = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    # Strategy 3: Volume ascending (small first)
    order3 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=False)
    
    # Strategy 4: Surface area descending
    order4 = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), _box_volume(boxes[i])), reverse=True)
    
    strategies = [order1, order2, order3, order4]
    
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    # Allocate time: ~50s for initial packing, rest for local search
    initial_deadline = start_time + 50.0
    
    for order in strategies:
        if time.time() > initial_deadline:
            break
        
        bins_try, result_try = _pack_with_order(order)
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
