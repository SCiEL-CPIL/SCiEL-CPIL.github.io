"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    start = time.time()
    deadline = start + 125.0  # Safety margin before 150s kill
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort by volume descending, with tie-breaking by max dimension
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (t[1][0]*t[1][1]*t[1][2], 
                                 max(t[1][0], t[1][1], t[1][2])), 
                 reverse=True)
    
    bins = []  # list of lists of placements per bin
    result = [None] * n
    bin_vols = []  # track volume per bin for fast sorting
    
    def _try_place(box, placements, W, H, D):
        """Find best position using a combined heuristic:
        - Primary: minimize the 'gap' (space not filled) to pack tightly
        - Secondary: prefer lower z, then lower y, then lower x
        Uses candidate positions from existing box boundaries."""
        w, h, d = box
        if w > W or h > H or d > D:
            return None
        
        # Candidate positions from boundaries
        xs = {0}
        ys = {0}
        zs = {0}
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
        
        # Precompute for faster overlap checks
        placed_boxes = [(x, y, z, bw, bh, bd) for (_, x, y, z, bw, bh, bd) in placements]
        
        for z in zs:
            for y in ys:
                for x in xs:
                    # Check overlap with all placed boxes
                    ok = True
                    for (ex, ey, ez, ew, eh, ed) in placed_boxes:
                        if x < ex + ew and ex < x + w and \
                           y < ey + eh and ey < y + h and \
                           z < ez + ed and ez < z + d:
                            ok = False
                            break
                    if ok:
                        # Combined scoring:
                        # 1. Minimize the maximum remaining dimension (tight fit)
                        rem_x = W - (x + w)
                        rem_y = H - (y + h)
                        rem_z = D - (z + d)
                        max_rem = max(rem_x, rem_y, rem_z)
                        
                        # 2. Prefer positions that are "lower" in the bin
                        #    to allow future boxes to be placed on top
                        pos_score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        
                        # Combined: prioritize tight fit, then position
                        # Use a weighted combination
                        score = max_rem * 1000 + pos_score
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for orig_idx, box in indexed:
        w, h, d = box
        placed = False
        
        # Try existing bins - sort by how full they are (most full first)
        if bins:
            # Sort bins by volume (most filled first) for best-fit
            bin_order = sorted(range(len(bins)), key=lambda bi: bin_vols[bi], reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline:
                    break
                pos = _try_place(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((bi, x, y, z, w, h, d))
                    bin_vols[bi] += w * h * d
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            if time.time() > deadline:
                # Emergency: place remaining boxes in new bins
                bi = len(bins)
                for oi, ob in indexed:
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
    
    # Local search: try to consolidate boxes into fewer bins
    # Multiple passes to allow cascading moves
    if time.time() < deadline - 15:
        max_passes = 3
        for pass_num in range(max_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            # Process boxes from last bin to first to enable cascading
            for orig_idx in range(n - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                box = boxes[orig_idx]
                w, h, d = box
                current_bin = result[orig_idx][0]
                if current_bin == 0:
                    continue
                
                # Try to move to an earlier bin
                moved = False
                for target_bin in range(current_bin):
                    if time.time() > deadline - 10:
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
                    pos = _try_place(box, target_placements, W, H, D)
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
    
    # Ensure all boxes have a placement
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            bin_vols.append(w * h * d)
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
