"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def _fits_in_bin(box, placements, W, H, D):
    """Try to place box (w,h,d) in a bin given existing placements.
    Returns (x,y,z) or None. Uses extreme point candidates."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Build candidate positions from existing box boundaries
    xs = [0]
    ys = [0]
    zs = [0]
    for (b, x, y, z, bw, bh, bd) in placements:
        if x + bw <= W:
            xs.append(x + bw)
        if y + bh <= H:
            ys.append(y + bh)
        if z + bd <= D:
            zs.append(z + bd)
    
    # Also add positions aligned to back of existing boxes (for better packing)
    for (b, x, y, z, bw, bh, bd) in placements:
        # Try placing against the "back" of each existing box
        if x >= w:
            xs.append(x - w)
        if y >= h:
            ys.append(y - h)
        if z >= d:
            zs.append(z - d)
    
    xs = sorted(set(x for x in xs if 0 <= x and x + w <= W))
    ys = sorted(set(y for y in ys if 0 <= y and y + h <= H))
    zs = sorted(set(z for z in zs if 0 <= z and z + d <= D))
    
    if not xs or not ys or not zs:
        return None
    
    # Sort candidates to prefer bottom-left-front (more compact)
    # We'll iterate and pick the first valid position
    for z in zs:
        for y in ys:
            for x in xs:
                ok = True
                for (b, ex, ey, ez, ew, eh, ed) in placements:
                    if x < ex + ew and ex < x + w and \
                       y < ey + eh and ey < y + h and \
                       z < ez + ed and ez < z + d:
                        ok = False
                        break
                if ok:
                    return (x, y, z)
    return None

def _fits_in_bin_best_fit(box, placements, W, H, D):
    """Try to place box in a bin, returning the position that minimizes 
    the resulting bounding box volume (best-fit). Returns (x,y,z) or None."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Build candidate positions
    xs = [0]
    ys = [0]
    zs = [0]
    for (b, x, y, z, bw, bh, bd) in placements:
        if x + bw <= W:
            xs.append(x + bw)
        if y + bh <= H:
            ys.append(y + bh)
        if z + bd <= D:
            zs.append(z + bd)
    
    xs = sorted(set(x for x in xs if 0 <= x and x + w <= W))
    ys = sorted(set(y for y in ys if 0 <= y and y + h <= H))
    zs = sorted(set(z for z in zs if 0 <= z and z + d <= D))
    
    if not xs or not ys or not zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for z in zs:
        for y in ys:
            for x in xs:
                ok = True
                for (b, ex, ey, ez, ew, eh, ed) in placements:
                    if x < ex + ew and ex < x + w and \
                       y < ey + eh and ey < y + h and \
                       z < ez + ed and ez < z + d:
                        ok = False
                        break
                if ok:
                    # Score: prefer positions that keep bounding box small
                    # and are close to origin (compact)
                    max_x = max(x + w, max((p[1] + p[4]) for p in placements), default=0)
                    max_y = max(y + h, max((p[2] + p[5]) for p in placements), default=0)
                    max_z = max(z + d, max((p[3] + p[6]) for p in placements), default=0)
                    score = max_x * max_y * max_z
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort by volume descending, with tie-breaking by max dimension
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], max(t[1])), reverse=True)
    
    bins = []  # list of lists of placements per bin
    result = [None] * n
    
    for orig_idx, box in indexed:
        w, h, d = box
        placed = False
        
        # Try existing bins with best-fit
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for bi in range(len(bins)):
            pos = _fits_in_bin_best_fit(box, bins[bi], W, H, D)
            if pos is not None:
                x, y, z = pos
                # Score: prefer bins where this placement results in smallest bounding box
                max_x = max(x + w, max((p[1] + p[4]) for p in bins[bi]), default=0)
                max_y = max(y + h, max((p[2] + p[5]) for p in bins[bi]), default=0)
                max_z = max(z + d, max((p[3] + p[6]) for p in bins[bi]), default=0)
                score = max_x * max_y * max_z
                if score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((best_bin, x, y, z, w, h, d))
            result[orig_idx] = (best_bin, x, y, z, w, h, d)
            placed = True
        
        if not placed:
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
        
        if time.time() > deadline:
            # Fill remaining with new bins
            bi = len(bins)
            for oi, ob in indexed:
                if result[oi] is None:
                    bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                    result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                    bi += 1
            break
    
    return result
# EVOLVE-BLOCK-END
