import time
import sys

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _can_place(box, pos, placements, W, H, D):
    x, y, z = pos
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position_bfd(box, placements, W, H, D):
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
        
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    return (x, y, z)
                    
    return None

def _find_position_best_fit(box, placements, W, H, D):
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
                    # Best fit: minimize sum of distances to nearest surfaces/boxes
                    # This encourages tighter packing
                    dist_x = min(x, W - (x + w))
                    dist_y = min(y, H - (y + h))
                    dist_z = min(z, D - (z + d))
                    
                    # Also consider proximity to existing boxes
                    min_dist_to_box = float('inf')
                    for (px, py, pz, pw, ph, pd) in placements:
                        dx = 0
                        if x + w <= px:
                            dx = px - (x + w)
                        elif x >= px + pw:
                            dx = x - (px + pw)
                        
                        dy = 0
                        if y + h <= py:
                            dy = py - (y + h)
                        elif y >= py + ph:
                            dy = y - (py + ph)
                        
                        dz = 0
                        if z + d <= pz:
                            dz = pz - (z + d)
                        elif z >= pz + pd:
                            dz = z - (pz + pd)
                        
                        dist = max(dx, dy, dz)
                        if dist < min_dist_to_box:
                            min_dist_to_box = dist
                        if min_dist_to_box == 0:
                            break
                    
                    # Score: prefer positions close to walls and boxes
                    score = dist_x + dist_y + dist_z + min_dist_to_box
                    
                    # Tie-break with BFL
                    score += z * 0.001 + y * 0.0001 + x * 0.00001
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    def _pack_with_order(order, finder_func):
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
        
        for i in range(n):
            if result[i] is None:
                bi = len(bins)
                w, h, d = boxes[i]
                bins.append([(0, 0, 0, w, h, d)])
                result[i] = (bi, 0, 0, 0, w, h, d)
        
        return bins, result
    
    def _score_solution(bins, result):
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
    
    # Generate multiple ordering strategies
    strategies = []
    
    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: _box_volume(boxes[i]), reverse=True)
    strategies.append((order1, _find_position_bfd))
    
    # Strategy 2: Max dimension descending
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append((order2, _find_position_bfd))
    
    # Strategy 3: Volume descending with best fit
    order3 = sorted(range(n), key=lambda i: _box_volume(boxes[i]), reverse=True)
    strategies.append((order3, _find_position_best_fit))
    
    # Strategy 4: Max dimension descending with best fit
    order4 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append((order4, _find_position_best_fit))
    
    # Strategy 5: Surface area descending
    order5 = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append((order5, _find_position_bfd))
    
    # Strategy 6: Surface area descending with best fit
    order6 = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append((order6, _find_position_best_fit))
    
    # Try all strategies and pick the best
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    for order, finder in strategies:
        if time.time() > deadline - 60.0:
            break
        
        bins_try, result_try = _pack_with_order(order, finder)
        score_try = _score_solution(bins_try, result_try)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_result = result_try
    
    bins = best_bins
    result = best_result
    
    # Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 3.0:
        last_bin_id = max(r[0] for r in result)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # --- Attempt Moves ---
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                if result[idx][0] == last_bin_id:
                    continue
                    
                cur_bin_id = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                last_placements = bins[last_bin_id]
                pos = _find_position_bfd(box, last_placements, W, H, D)
                
                if pos is not None:
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
            if not improved and time.time() < deadline - 10.0:
                last_boxes_indices = [i for i in range(n) if result[i][0] == last_bin_id]
                
                for idx_last in last_boxes_indices:
                    if time.time() > deadline - 10.0:
                        break
                    
                    box_last = boxes[idx_last]
                    wl, hl, dl = box_last
                    pos_last = (result[idx_last][1], result[idx_last][2], result[idx_last][3])
                    target_last_placement = (pos_last[0], pos_last[1], pos_last[2], wl, hl, dl)
                    
                    for idx_other in range(n):
                        if time.time() > deadline - 10.0:
                            break
                        if result[idx_other][0] == last_bin_id:
                            continue
                            
                        box_other = boxes[idx_other]
                        wo, ho, do = box_other
                        pos_other = (result[idx_other][1], result[idx_other][2], result[idx_other][3])
                        target_other_placement = (pos_other[0], pos_other[1], pos_other[2], wo, ho, do)
                        
                        if _box_volume(box_other) <= _box_volume(box_last):
                            continue
                            
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
                            
                        bins[last_bin_id] = last_placements_temp + [(pos_new[0], pos_new[1], pos_new[2], wo, ho, do)]
                        bins[other_bin_id] = other_placements_temp + [(pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)]
                        
                        result[idx_last] = (other_bin_id, pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)
                        result[idx_other] = (last_bin_id, pos_new[0], pos_new[1], pos_new[2], wo, ho, do)
                        
                        improved = True
                        break
                    if improved:
                        break
                
    # Phase 3: Try to reduce bin count by consolidating bins
    if time.time() < deadline - 5.0:
        bin_ids = sorted(set(r[0] for r in result))
        
        for target_bin in bin_ids:
            if time.time() > deadline - 5.0:
                break
            
            if target_bin == max(bin_ids):
                continue
            
            boxes_in_bin = [i for i in range(n) if result[i][0] == target_bin]
            if not boxes_in_bin:
                continue
            
            success = True
            temp_result = result[:]
            temp_bins = {bi: bins[bi][:] for bi in bin_ids}
            
            for idx in boxes_in_bin:
                box = boxes[idx]
                w, h, d = box
                placed = False
                
                for other_bin in bin_ids:
                    if other_bin == target_bin:
                        continue
                    pos = _find_position_bfd(box, temp_bins[other_bin], W, H, D)
                    if pos is not None:
                        temp_bins[other_bin].append((pos[0], pos[1], pos[2], w, h, d))
                        temp_result[idx] = (other_bin, pos[0], pos[1], pos[2], w, h, d)
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                result = temp_result
                for bi in bin_ids:
                    if bi != target_bin:
                        bins[bi] = temp_bins[bi]
                if target_bin in bins:
                    del bins[target_bin]
    
    # Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
            
    return result
