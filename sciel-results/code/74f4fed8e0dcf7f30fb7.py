"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _can_place(placements, W, H, D, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any placement."""
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        if x < ex + ew and ex < x + w and \
           y < ey + eh and ey < y + h and \
           z < ez + ed and ez < z + d:
            return False
    return True

def _find_position(placements, W, H, D, w, h, d, strategy='first_fit'):
    """Find a valid position for a box. Returns (x,y,z) or None."""
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate positions from existing box boundaries + 0
    xs = {0}
    ys = {0}
    zs = {0}
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        xs.add(ex + ew)
        ys.add(ey + eh)
        zs.add(ez + ed)
    
    # Also add positions that are exactly w, h, d away from right/back/top
    xs.add(max(0, W - w))
    ys.add(max(0, H - h))
    zs.add(max(0, D - d))
    
    valid_xs = sorted([x for x in xs if 0 <= x and x + w <= W])
    valid_ys = sorted([y for y in ys if 0 <= y and y + h <= H])
    valid_zs = sorted([z for z in zs if 0 <= z and z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if not _can_place(placements, W, H, D, x, y, z, w, h, d):
                    continue
                
                if strategy == 'first_fit':
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                elif strategy == 'best_fit':
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    score = max(rem_x, rem_y, rem_z) * 1000000 + rem_x * 1000 + rem_y * 100 + rem_z
                elif strategy == 'tightest':
                    # Minimize the sum of remaining dimensions (tightest fit)
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    score = (rem_x + rem_y + rem_z) * 1000000 + z * 1000 + y * 10 + x
                else:
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
    
    return best_pos

def _bin_volume(placements):
    """Calculate total volume of boxes in a bin."""
    return sum(bw * bh * bd for (_, _, _, _, bw, bh, bd) in placements)

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes: volume descending, then largest dimension descending
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], 
                                 max(t[1][0], t[1][1], t[1][2])), 
                 reverse=True)
    
    bins = []
    result = [None] * n
    
    # Phase 1: Initial placement with best-fit decreasing
    for idx, (orig_idx, box) in enumerate(indexed):
        if time.time() > deadline - 5:
            # Emergency placement
            bi = len(bins)
            for oi, ob in indexed[idx:]:
                if result[oi] is None:
                    bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                    result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                    bi += 1
            break
        
        w, h, d = box
        placed = False
        
        # Try existing bins, most full first
        if bins:
            bin_order = sorted(range(len(bins)), 
                               key=lambda bi: _bin_volume(bins[bi]), 
                               reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline - 3:
                    break
                pos = _find_position(bins[bi], W, H, D, w, h, d, strategy='first_fit')
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Ensure all boxes placed
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Consolidation - move boxes to reduce bin count
    if time.time() < deadline - 15:
        for pass_num in range(5):
            if time.time() > deadline - 5:
                break
            
            moved_any = False
            for orig_idx in range(n):
                if time.time() > deadline - 5:
                    break
                
                w, h, d = boxes[orig_idx]
                current_bin = result[orig_idx][0]
                
                if current_bin == 0:
                    continue
                
                # Find this box's index in its bin
                cur_placements = bins[current_bin]
                box_idx = None
                for i, p in enumerate(cur_placements):
                    if p[1] == result[orig_idx][1] and \
                       p[2] == result[orig_idx][2] and p[3] == result[orig_idx][3] and \
                       p[4] == w and p[5] == h and p[6] == d:
                        box_idx = i
                        break
                if box_idx is None:
                    continue
                
                # Try moving to each earlier bin
                for target_bin in range(current_bin):
                    if time.time() > deadline - 5:
                        break
                    
                    cur_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                    target_placements = bins[target_bin]
                    
                    pos = _find_position(target_placements, W, H, D, w, h, d, strategy='first_fit')
                    if pos is not None:
                        x, y, z = pos
                        bins[target_bin].append((target_bin, x, y, z, w, h, d))
                        bins[current_bin] = cur_without
                        result[orig_idx] = (target_bin, x, y, z, w, h, d)
                        moved_any = True
                        break
                
                if moved_any:
                    # Restart consolidation after a move
                    break
            
            if not moved_any:
                break
    
    # Phase 3: Optimize last bin fill
    if time.time() < deadline - 5:
        if not result:
            return result
        
        last_bin = max(r[0] for r in result)
        
        # Try moving boxes from other bins to last bin to improve fill
        for orig_idx in range(n):
            if time.time() > deadline - 3:
                break
            
            current_bin = result[orig_idx][0]
            if current_bin == last_bin:
                continue
            
            w, h, d = boxes[orig_idx]
            cur_placements = bins[current_bin]
            box_idx = None
            for i, p in enumerate(cur_placements):
                if p[1] == result[orig_idx][1] and \
                   p[2] == result[orig_idx][2] and p[3] == result[orig_idx][3] and \
                   p[4] == w and p[5] == h and p[6] == d:
                    box_idx = i
                    break
            if box_idx is None:
                continue
            
            cur_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
            last_placements = bins[last_bin]
            
            pos = _find_position(last_placements, W, H, D, w, h, d, strategy='tightest')
            if pos is not None:
                x, y, z = pos
                bins[last_bin].append((last_bin, x, y, z, w, h, d))
                bins[current_bin] = cur_without
                result[orig_idx] = (last_bin, x, y, z, w, h, d)
    
    # Final: reassign consecutive bin labels
    if time.time() < deadline - 2:
        bin_to_label = {}
        new_label = 0
        for bi in range(len(bins)):
            if bins[bi]:
                bin_to_label[bi] = new_label
                new_label += 1
        
        if bin_to_label:
            for i in range(n):
                old_bin = result[i][0]
                new_bin = bin_to_label.get(old_bin, 0)
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                result[i][4], result[i][5], result[i][6])
    
    return result
# EVOLVE-BLOCK-END
