"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _fits_in_bin(box, placements, W, H, D):
    """Try to place box (w,h,d) in a bin given existing placements.
    Returns (x,y,z) or None. Uses best-fit-decreasing heuristic with
    candidate positions from existing box boundaries."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Candidate positions derived from boundaries of existing boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (_, x, y, z, bw, bh, bd) in placements:
        xs.add(x + bw)
        ys.add(y + bh)
        zs.add(z + bd)
    
    # Filter to valid positions
    xs = sorted([x for x in xs if x + w <= W])
    ys = sorted([y for y in ys if y + h <= H])
    zs = sorted([z for z in zs if z + d <= D])
    
    if not xs or not ys or not zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for z in zs:
        for y in ys:
            for x in xs:
                # Check overlap
                ok = True
                for (_, ex, ey, ez, ew, eh, ed) in placements:
                    if x < ex + ew and ex < x + w and \
                       y < ey + eh and ey < y + h and \
                       z < ez + ed and ez < z + d:
                        ok = False
                        break
                if ok:
                    # Score: prefer positions that minimize wasted space
                    # Use the "lowest, then leftmost, then backmost" heuristic
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def _fits_in_bin_bestfit(box, placements, W, H, D):
    """Try to place box using best-fit: minimize the remaining space in the bin.
    Returns (x,y,z) or None."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Candidate positions derived from boundaries of existing boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (_, x, y, z, bw, bh, bd) in placements:
        xs.add(x + bw)
        ys.add(y + bh)
        zs.add(z + bd)
    
    # Filter to valid positions
    xs = sorted([x for x in xs if x + w <= W])
    ys = sorted([y for y in ys if y + h <= H])
    zs = sorted([z for z in zs if z + d <= D])
    
    if not xs or not ys or not zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for z in zs:
        for y in ys:
            for x in xs:
                # Check overlap
                ok = True
                for (_, ex, ey, ez, ew, eh, ed) in placements:
                    if x < ex + ew and ex < x + w and \
                       y < ey + eh and ey < y + h and \
                       z < ez + ed and ez < z + d:
                        ok = False
                        break
                if ok:
                    # Best-fit: minimize the "hole" left behind
                    # Score by how tightly the box fits (smaller max remaining dimension)
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    score = max(rem_x, rem_y, rem_z)
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    n = len(boxes)
    
    # Sort by volume descending, with tie-breaking by max dimension
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], 
                                 max(t[1][0], t[1][1], t[1][2])), 
                 reverse=True)
    
    bins = []  # list of lists of placements per bin
    result = [None] * n
    
    def _try_place(box, placements, W, H, D):
        """Try multiple heuristics to find a position."""
        w, h, d = box
        if w > W or h > H or d > D:
            return None
        
        # Candidate positions
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (_, x, y, z, bw, bh, bd) in placements:
            xs.add(x + bw)
            ys.add(y + bh)
            zs.add(z + bd)
        
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        if not xs or not ys or not zs:
            return None
        
        best_pos = None
        best_score = float('inf')
        
        for z in zs:
            for y in ys:
                for x in xs:
                    # Check overlap
                    ok = True
                    for (_, ex, ey, ez, ew, eh, ed) in placements:
                        if x < ex + ew and ex < x + w and \
                           y < ey + eh and ey < y + h and \
                           z < ez + ed and ez < z + d:
                            ok = False
                            break
                    if ok:
                        # Combined score: prefer low z, low y, low x, but also
                        # consider how full the bin gets
                        score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for orig_idx, box in indexed:
        w, h, d = box
        placed = False
        
        # Try existing bins - best fit decreasing (try bins with most filled first)
        if bins:
            # Sort bins by how full they are (most full first)
            def _bin_fill(bi):
                vol = sum(bw*bh*bd for (_, _, _, _, bw, bh, bd) in bins[bi])
                return vol / (W * H * D)
            
            bin_order = sorted(range(len(bins)), key=_bin_fill, reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline:
                    break
                pos = _try_place(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            if time.time() > deadline:
                # Fill remaining with new bins
                bi = len(bins)
                for oi, ob in indexed:
                    if result[oi] is None:
                        bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                        result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                        bi += 1
                break
            
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Final optimization: try to improve by moving boxes to reduce bin count
    # This is a simple local search: try to move each box to an earlier bin
    if time.time() < deadline - 10:
        # Try to consolidate: for each box, see if it can fit in an earlier bin
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
                    if p[0] == current_bin and p[1] == result[orig_idx][1] and \
                       p[2] == result[orig_idx][2] and p[3] == result[orig_idx][3]:
                        box_idx = i
                        break
                if box_idx is None:
                    continue
                
                cur_placements_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                target_placements = bins[target_bin]
                
                # Check if it fits in target bin
                pos = _try_place(box, target_placements, W, H, D)
                if pos is not None:
                    # Move it
                    x, y, z = pos
                    bins[target_bin].append((target_bin, x, y, z, w, h, d))
                    bins[current_bin] = cur_placements_without
                    result[orig_idx] = (target_bin, x, y, z, w, h, d)
                    moved = True
                    break
            
            if moved:
                continue
    
    # Ensure all boxes have a placement
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
