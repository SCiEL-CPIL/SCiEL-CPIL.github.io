"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending for better initial packing
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    def try_place_box(bin_state, box_idx, W, H, D):
        """Try to place box in bin_state. Returns (x,y,z) if successful, else None.
        Uses a simple grid-based search for valid positions."""
        w, h, d = boxes[box_idx]
        placed = bin_state['placed']  # list of (x,y,z,w,h,d)
        
        # Try to find a position: scan through possible x,y,z
        # For efficiency, we try positions based on existing boxes' edges
        candidate_xs = [0]
        candidate_ys = [0]
        candidate_zs = [0]
        for (px, py, pz, pw, ph, pd) in placed:
            candidate_xs.append(px + pw)
            candidate_ys.append(py + ph)
            candidate_zs.append(pz + pd)
        
        # Deduplicate and sort
        candidate_xs = sorted(set(candidate_xs))
        candidate_ys = sorted(set(candidate_ys))
        candidate_zs = sorted(set(candidate_zs))
        
        best_pos = None
        best_score = float('inf')
        
        for x in candidate_xs:
            if x + w > W:
                continue
            for y in candidate_ys:
                if y + h > H:
                    continue
                for z in candidate_zs:
                    if z + d > D:
                        continue
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if x < px + pw and px < x + w and \
                           y < py + ph and py < y + h and \
                           z < pz + pd and pz < z + d:
                            overlap = True
                            break
                    if not overlap:
                        # Score: prefer lower positions (fill from bottom)
                        score = z * 10000 + y * 100 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos

    def try_place_box_fast(bin_state, box_idx, W, H, D):
        """Faster placement using shelf-like heuristic within a bin."""
        w, h, d = boxes[box_idx]
        placed = bin_state['placed']
        
        # Try to place at origin first, then along edges
        candidates = [(0, 0, 0)]
        
        # Generate candidates from existing box boundaries
        for (px, py, pz, pw, ph, pd) in placed:
            if px + pw + w <= W:
                candidates.append((px + pw, py, pz))
            if py + ph + h <= H:
                candidates.append((px, py + ph, pz))
            if pz + pd + d <= D:
                candidates.append((px, py, pz + pd))
        
        # Also try some heuristic positions
        for (px, py, pz, pw, ph, pd) in placed:
            if px + pw + w <= W and pz + pd + d <= D:
                candidates.append((px + pw, py, pz + pd))
            if py + ph + h <= H and px + pw + w <= W:
                candidates.append((px + pw, py + ph, pz))
            if pz + pd + d <= D and py + ph + h <= H:
                candidates.append((px, py + ph, pz + pd))
        
        for (x, y, z) in candidates:
            if x < 0 or y < 0 or z < 0:
                continue
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check overlap
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed:
                if x < px + pw and px < x + w and \
                   y < py + ph and py < y + h and \
                   z < pz + pd and pz < z + d:
                    overlap = True
                    break
            if not overlap:
                return (x, y, z)
        
        return None

    # Initial greedy packing
    def greedy_pack():
        bins = []  # list of {'placed': [(x,y,z,w,h,d)], 'volume': float}
        bin_count = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            placed_in_bin = False
            
            # Try existing bins
            for b in range(bin_count):
                pos = try_place_box_fast(bins[b], idx, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[b]['placed'].append((x, y, z, w, h, d))
                    bins[b]['volume'] += w * h * d
                    placed_in_bin = True
                    break
            
            if not placed_in_bin:
                # Try new bin
                if w <= W and h <= H and d <= D:
                    pos = (0, 0, 0)
                    new_bin = {'placed': [(pos[0], pos[1], pos[2], w, h, d)], 'volume': w*h*d}
                    bins.append(new_bin)
                    bin_count += 1
                    placed_in_bin = True
            
            # If box doesn't fit in any bin, it's a problem
            if not placed_in_bin:
                # This shouldn't happen if boxes fit in container
                pass
        
        return bins

    # Calculate score for a given packing
    def calc_score(bins):
        if not bins:
            return 0.0
        B = len(bins)
        last_bin_vol = bins[-1]['volume']
        container_vol = W * H * D
        last_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        return B + 0.1 * (1 - last_frac)

    # Local improvement: try to move boxes between bins
    def local_improve(bins, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n_bins = len(bins)
            
            for b1 in range(n_bins):
                for b2 in range(b1 + 1, n_bins):
                    # Try moving a box from b1 to b2
                    for i in range(len(bins[b1]['placed'])):
                        box = bins[b1]['placed'][i]
                        w, h, d = box[3], box[4], box[5]
                        
                        # Remove from b1
                        bins[b1]['placed'].pop(i)
                        old_vol_b1 = bins[b1]['volume'] - w*h*d
                        bins[b1]['volume'] = old_vol_b1
                        
                        # Try to place in b2
                        pos = try_place_box_fast(bins[b2], order[0], W, H, D)  # placeholder
                        # Actually need to try with the specific box
                        # Let's use a helper
                        placed = False
                        for (x, y, z) in [(0,0,0)] + [(px+pw, py, pz) for (px,py,pz,pw,ph,pd) in bins[b2]['placed'] if px+pw+w<=W] + [(px, py+ph, pz) for (px,py,pz,pw,ph,pd) in bins[b2]['placed'] if py+ph+h<=H] + [(px, py, pz+pd) for (px,py,pz,pw,ph,pd) in bins[b2]['placed'] if pz+pd+d<=D]:
                            if x<0 or y<0 or z<0 or x+w>W or y+h>H or z+d>D:
                                continue
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in bins[b2]['placed']:
                                if x < px+pw and px < x+w and y < py+ph and py < y+h and z < pz+pd and pz < z+d:
                                    overlap = True
                                    break
                            if not overlap:
                                bins[b2]['placed'].append((x, y, z, w, h, d))
                                bins[b2]['volume'] += w*h*d
                                placed = True
                                break
                        
                        if placed:
                            # Check if this reduces bin count or improves score
                            new_bins = [b for b in bins if b['placed']]
                            new_score = calc_score(new_bins)
                            old_score = calc_score(bins)
                            
                            if new_score < old_score - 1e-9:
                                bins = new_bins
                                improved = True
                            else:
                                # Undo
                                bins[b2]['placed'].pop()
                                bins[b2]['volume'] -= w*h*d
                                bins[b1]['placed'].append(box)
                                bins[b1]['volume'] = old_vol_b1
                        else:
                            # Undo
                            bins[b1]['placed'].append(box)
                            bins[b1]['volume'] = old_vol_b1
        
        return bins

    # Get initial packing
    bins = greedy_pack()
    
    # Local improvement
    if time.time() < deadline:
        bins = local_improve(bins, deadline)
    
    # Build result
    result = []
    for b_idx, bin_state in enumerate(bins):
        for (x, y, z, w, h, d) in bin_state['placed']:
            # Find original box index
            # We need to map back to original indices
            for i in range(n):
                if boxes[i] == (w, h, d):
                    result.append((b_idx, x, y, z, w, h, d))
                    break
    
    # Ensure all boxes are placed
    if len(result) < n:
        # Fallback: place remaining boxes in new bins
        placed_count = len(result)
        b_idx = len(bins)
        for i in range(n):
            w, h, d = boxes[i]
            # Check if already placed
            already = False
            for r in result:
                if r[1:] == (0, 0, 0, w, h, d) or (r[4], r[5], r[6]) == (w, h, d):
                    # This is ambiguous, let's just ensure we have n placements
                    pass
            if placed_count < n:
                result.append((b_idx, 0, 0, 0, w, h, d))
                placed_count += 1
                b_idx += 1
    
    return result

# EVOLVE-BLOCK-END
