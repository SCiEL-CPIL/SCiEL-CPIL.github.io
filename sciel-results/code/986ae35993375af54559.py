"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _candidate_positions(placements, W, H, D):
    """Generate candidate x, y, z positions from box boundaries."""
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (_, x, y, z, bw, bh, bd) in placements:
        xs.add(x + bw)
        ys.add(y + bh)
        zs.add(z + bd)
    
    xs = sorted([x for x in xs if x <= W])
    ys = sorted([y for y in ys if y <= H])
    zs = sorted([z for z in zs if z <= D])
    return xs, ys, zs

def _try_place_box(box, placements, W, H, D, strategy='first_fit'):
    """Try to place a box in a bin. Returns (x,y,z) or None.
    
    strategy: 'first_fit' - lowest, then leftmost, then backmost
              'best_fit' - minimize remaining space
              'maximize_fill' - maximize contact with existing boxes (for last bin filling)
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs, ys, zs = _candidate_positions(placements, W, H, D)
    
    if not xs or not ys or not zs:
        return None
    
    # Filter candidates that fit
    valid_xs = [x for x in xs if x + w <= W]
    valid_ys = [y for y in ys if y + h <= H]
    valid_zs = [z for z in zs if z + d <= D]
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    n_placements = len(placements)
    
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                # Check overlap
                ok = True
                for (_, ex, ey, ez, ew, eh, ed) in placements:
                    if x < ex + ew and ex < x + w and \
                       y < ey + eh and ey < y + h and \
                       z < ez + ed and ez < z + d:
                        ok = False
                        break
                if not ok:
                    continue
                
                if strategy == 'first_fit':
                    # Prefer lowest z, then lowest y, then lowest x
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                elif strategy == 'best_fit':
                    # Minimize the maximum remaining dimension
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    score = max(rem_x, rem_y, rem_z) * 1000000 + rem_x * 1000 + rem_y * 100 + rem_z
                elif strategy == 'maximize_fill':
                    # For last bin: prefer positions that fill the bin tightly
                    # Minimize distance from bottom, and maximize how much of the
                    # remaining space is "used up" by this box
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    # We want to minimize remaining volume
                    score = rem_x * rem_y * rem_z + z * 1000000 + y * 1000 + x
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], 
                                 max(t[1][0], t[1][1], t[1][2])), 
                 reverse=True)
    
    bins = []  # list of lists of placements per bin
    result = [None] * n
    
    # Phase 1: Place all boxes using best-fit decreasing
    # Use first-fit for most boxes, but for the last few boxes, use maximize_fill
    # to ensure the last bin is well-filled
    
    for phase_idx, (orig_idx, box) in enumerate(indexed):
        if time.time() > deadline - 5:
            # Emergency: place remaining boxes in new bins at origin
            bi = len(bins)
            for oi, ob in indexed[phase_idx:]:
                if result[oi] is None:
                    bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                    result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                    bi += 1
            break
        
        w, h, d = box
        placed = False
        
        # Determine strategy based on how many boxes remain
        # For the last ~10 boxes, use maximize_fill to pack the last bin tightly
        remaining = n - phase_idx
        if remaining <= 10 and bins:
            strategy = 'maximize_fill'
        elif remaining <= 20 and bins:
            strategy = 'best_fit'
        else:
            strategy = 'first_fit'
        
        # Try existing bins
        if bins:
            # Sort bins: for first_fit, try most full first (best-fit decreasing)
            # For maximize_fill, try the last bin first (to fill it up)
            if strategy == 'maximize_fill':
                # Try the last bin first, then others
                bin_order = [len(bins) - 1] + list(range(len(bins) - 1))
            else:
                # Try most full bins first
                bin_order = sorted(range(len(bins)), 
                                   key=lambda bi: _bin_volume(bins[bi]), 
                                   reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline - 3:
                    break
                pos = _try_place_box(box, bins[bi], W, H, D, strategy=strategy)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            if time.time() > deadline - 3:
                # Emergency
                bi = len(bins)
                for oi, ob in indexed[phase_idx:]:
                    if result[oi] is None:
                        bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                        result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                        bi += 1
                break
            
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Ensure all boxes have a placement
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Local search - try to move boxes to reduce bin count
    # and improve last bin fill
    if time.time() < deadline - 10:
        # Multiple passes of consolidation
        for pass_num in range(3):
            if time.time() > deadline - 5:
                break
            
            # Sort boxes by original index for consistent iteration
            for orig_idx in range(n):
                if time.time() > deadline - 5:
                    break
                
                box = boxes[orig_idx]
                w, h, d = box
                current_bin = result[orig_idx][0]
                
                if current_bin == 0:
                    continue
                
                # Try to move to an earlier bin
                moved = False
                for target_bin in range(current_bin):
                    if time.time() > deadline - 5:
                        break
                    
                    # Temporarily remove from current bin
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
                    target_placements = bins[target_bin]
                    
                    # Check if it fits in target bin
                    pos = _try_place_box(box, target_placements, W, H, D, strategy='first_fit')
                    if pos is not None:
                        # Move it
                        x, y, z = pos
                        bins[target_bin].append((target_bin, x, y, z, w, h, d))
                        bins[current_bin] = cur_without
                        result[orig_idx] = (target_bin, x, y, z, w, h, d)
                        moved = True
                        break
                
                if moved:
                    continue
    
    # Phase 3: Try to improve the last bin's fill ratio
    # Move boxes from earlier bins to the last bin if it improves the score
    if time.time() < deadline - 5:
        last_bin = max(r[0] for r in result)
        last_bin_placements = bins[last_bin]
        
        # Try to move boxes from other bins to the last bin
        for orig_idx in range(n):
            if time.time() > deadline - 3:
                break
            
            current_bin = result[orig_idx][0]
            if current_bin == last_bin:
                continue
            
            box = boxes[orig_idx]
            w, h, d = box
            
            # Only try to move if it would improve the last bin's fill
            # without increasing the bin count
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
            
            # Check if removing this box leaves the current bin empty
            # If so, it reduces bin count - good!
            # Or if it just moves to last bin - improves last bin fill
            
            pos = _try_place_box(box, last_bin_placements, W, H, D, strategy='maximize_fill')
            if pos is not None:
                x, y, z = pos
                bins[last_bin].append((last_bin, x, y, z, w, h, d))
                bins[current_bin] = cur_without
                result[orig_idx] = (last_bin, x, y, z, w, h, d)
    
    # Final cleanup: ensure no empty bins with high labels
    # Reassign bin labels to be consecutive
    if time.time() < deadline - 2:
        bin_to_label = {}
        new_label = 0
        for bi in range(len(bins)):
            if bins[bi]:  # non-empty
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
