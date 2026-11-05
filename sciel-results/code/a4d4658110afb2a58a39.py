"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _try_place_box(box, placements, W, H, D, strategy='best_fit'):
    """Try to place a box in a bin. Returns (x,y,z) or None.
    
    Uses a smarter candidate generation and scoring to find tight packings.
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Generate candidate positions from existing box boundaries
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        # Right side of each box
        if ex + ew <= W - w:
            xs.add(ex + ew)
        # Left side of each box (if box can fit to the left)
        if ex >= w:
            xs.add(ex - w)
        if ey + eh <= H - h:
            ys.add(ey + eh)
        if ey >= h:
            ys.add(ey - h)
        if ez + ed <= D - d:
            zs.add(ez + ed)
        if ez >= d:
            zs.add(ez - d)
    
    # Also add positions that align with the far walls
    xs.add(W - w)
    ys.add(H - h)
    zs.add(D - d)
    
    xs = sorted([x for x in xs if 0 <= x <= W - w])
    ys = sorted([y for y in ys if 0 <= y <= H - h])
    zs = sorted([z for z in zs if 0 <= z <= D - d])
    
    if not xs or not ys or not zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    # Limit search to keep it fast
    max_candidates = 200
    candidate_count = 0
    
    for z in zs:
        for y in ys:
            for x in xs:
                candidate_count += 1
                if candidate_count > max_candidates:
                    break
                
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
                
                # Score: minimize distance to walls (prefer positions close to corners/walls)
                # This creates compact packings
                dist_x = min(x, W - x - w)
                dist_y = min(y, H - y - h)
                dist_z = min(z, D - z - d)
                
                if strategy == 'best_fit':
                    # Prefer positions that minimize the "wasted space" around the box
                    # Lower is better
                    score = (dist_x + dist_y + dist_z) * 1000 + z * 100 + y * 10 + x
                elif strategy == 'maximize_fill':
                    # For last bin: prefer positions that leave minimal remaining volume
                    rem_x = W - x - w
                    rem_y = H - y - h
                    rem_z = D - z - d
                    score = rem_x * rem_y * rem_z + z * 1000000 + y * 1000 + x
                else:  # first_fit
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
            if candidate_count > max_candidates:
                break
        if candidate_count > max_candidates:
            break
    
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], 
                                 max(t[1][0], t[1][1], t[1][2])), 
                 reverse=True)
    
    bins = []  # list of lists of placements per bin
    result = [None] * n
    bin_vols = []  # track volume per bin for fast sorting
    
    # Phase 1: Place all boxes using best-fit decreasing
    for phase_idx, (orig_idx, box) in enumerate(indexed):
        if time.time() > deadline - 5:
            # Emergency: place remaining boxes in new bins at origin
            bi = len(bins)
            for oi, ob in indexed[phase_idx:]:
                if result[oi] is None:
                    bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                    bin_vols.append(ob[0] * ob[1] * ob[2])
                    result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                    bi += 1
            break
        
        w, h, d = box
        placed = False
        
        # Determine strategy based on how many boxes remain
        remaining = n - phase_idx
        if remaining <= 8 and bins:
            strategy = 'maximize_fill'
        else:
            strategy = 'best_fit'
        
        # Try existing bins, most full first
        if bins:
            bin_order = sorted(range(len(bins)), 
                               key=lambda bi: bin_vols[bi], 
                               reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline - 3:
                    break
                pos = _try_place_box(box, bins[bi], W, H, D, strategy=strategy)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    bin_vols[bi] += w * h * d
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
                        bin_vols.append(ob[0] * ob[1] * ob[2])
                        result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                        bi += 1
                break
            
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            bin_vols.append(w * h * d)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Ensure all boxes have a placement
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            bin_vols.append(w * h * d)
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Local search - try to move boxes to reduce bin count
    if time.time() < deadline - 10:
        for pass_num in range(6):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Process boxes from last bin to first to enable cascading
            for orig_idx in range(n - 1, -1, -1):
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
                    
                    # Find and remove from current bin
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
                    
                    cur_placements_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                    target_placements = bins[target_bin]
                    
                    # Check if it fits in target bin
                    pos = _try_place_box(box, target_placements, W, H, D, strategy='best_fit')
                    if pos is not None:
                        # Move it
                        x, y, z = pos
                        bins[target_bin].append((target_bin, x, y, z, w, h, d))
                        bins[current_bin] = cur_placements_without
                        bin_vols[target_bin] += w * h * d
                        bin_vols[current_bin] -= w * h * d
                        result[orig_idx] = (target_bin, x, y, z, w, h, d)
                        moved = True
                        improved = True
                        break
                
                if moved:
                    continue
            
            if not improved:
                break
    
    # Phase 3: Try to eliminate the last bin by moving its boxes to earlier bins
    if time.time() < deadline - 3 and bins:
        last_bin_idx = max(result[i][0] for i in range(n))
        
        if last_bin_idx > 0:
            last_bin_boxes = [i for i in range(n) if result[i][0] == last_bin_idx]
            last_bin_boxes.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            
            for orig_idx in last_bin_boxes:
                if time.time() > deadline - 3:
                    break
                
                box = boxes[orig_idx]
                w, h, d = box
                
                # Remove from last bin
                cur_placements = bins[last_bin_idx]
                box_idx = None
                for i, p in enumerate(cur_placements):
                    if p[1] == result[orig_idx][1] and \
                       p[2] == result[orig_idx][2] and p[3] == result[orig_idx][3] and \
                       p[4] == w and p[5] == h and p[6] == d:
                        box_idx = i
                        break
                if box_idx is None:
                    continue
                
                cur_placements_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                bins[last_bin_idx] = cur_placements_without
                bin_vols[last_bin_idx] -= w * h * d
                
                # Try to place in an earlier bin
                placed = False
                for target_bin in range(last_bin_idx):
                    if time.time() > deadline - 3:
                        break
                    pos = _try_place_box(box, bins[target_bin], W, H, D, strategy='best_fit')
                    if pos is not None:
                        x, y, z = pos
                        bins[target_bin].append((target_bin, x, y, z, w, h, d))
                        bin_vols[target_bin] += w * h * d
                        result[orig_idx] = (target_bin, x, y, z, w, h, d)
                        placed = True
                        break
                
                if not placed:
                    # Put it back
                    bins[last_bin_idx].append((last_bin_idx, result[orig_idx][1], result[orig_idx][2], result[orig_idx][3], w, h, d))
                    bin_vols[last_bin_idx] += w * h * d
    
    return result
# EVOLVE-BLOCK-END
