"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by largest dimension descending
    # We need to keep track of original index to return correct dimensions
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda ib: (-ib[1][0]*ib[1][1]*ib[1][2], -max(ib[1])))
    
    # Each bin maintains a list of placed boxes: (x, y, z, w, h, d)
    # We use a simple shelf-based approach per bin but with best-fit
    # For each bin, we try to place the box at the best position
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in W,H,D without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap: two boxes overlap if they overlap on all three axes
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in this bin.
        Uses a heuristic: try to place at the position that minimizes
        the 'wasted' space, preferring lower z, then lower y, then lower x.
        We sample candidate positions based on existing box faces."""
        # Generate candidate x, y, z positions
        # x candidates: 0 and right faces of existing boxes
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                x_candidates.append(px + pw)
            if py + ph <= H:
                y_candidates.append(py + ph)
            if pz + pd <= D:
                z_candidates.append(pz + pd)
        
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for z in z_candidates:
            if z + d > D:
                break
            for y in y_candidates:
                if y + h > H:
                    break
                for x in x_candidates:
                    if x + w > W:
                        break
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        # Also prefer positions that are "tight"
                        score = z * 10000 + y * 100 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Multi-start with different sorting strategies
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_placements = None
    best_score = float('inf')
    
    # Try multiple sorting strategies
    sort_strategies = [
        lambda boxes: sorted(boxes, key=lambda b: (-b[0]*b[1]*b[2], -max(b))),  # volume desc
        lambda boxes: sorted(boxes, key=lambda b: (-max(b), -b[0]*b[1]*b[2])),  # largest dim desc
        lambda boxes: sorted(boxes, key=lambda b: (-b[0]*b[1], -b[0]*b[2], -b[1]*b[2])),  # face area desc
        lambda boxes: sorted(boxes, key=lambda b: (-b[0]*b[1]*b[2], -b[0], -b[1], -b[2])),  # volume, then dims
    ]
    
    for strat_idx, sort_func in enumerate(sort_strategies):
        if time.time() > deadline:
            break
        
        sorted_boxes = sort_func(list(boxes))
        
        # Place boxes
        bins = []  # list of lists of (x, y, z, w, h, d)
        placements = []  # (bin_idx, x, y, z, w, h, d)
        
        for orig_idx, (w, h, d) in enumerate(sorted_boxes):
            placed = False
            
            # Try existing bins first (best-fit)
            best_bin = -1
            best_pos = None
            best_fit_score = float('inf')
            
            for bi, bin_boxes in enumerate(bins):
                pos = find_position(bin_boxes, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    # Fit score: prefer bins that are more filled, or use z+y+x as tiebreaker
                    # Use the position score as proxy
                    fit_score = z * 10000 + y * 100 + x
                    if fit_score < best_fit_score:
                        best_fit_score = fit_score
                        best_bin = bi
                        best_pos = pos
            
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements.append((best_bin, x, y, z, w, h, d))
                placed = True
            
            # If not placed, try new bins
            if not placed:
                # Try a few new bin positions
                for attempt in range(3):
                    if time.time() > deadline:
                        break
                    # New bin: try to place at origin first
                    pos = find_position([], w, h, d)
                    if pos is not None:
                        x, y, z = pos
                        new_bin = [(x, y, z, w, h, d)]
                        bins.append(new_bin)
                        bi = len(bins) - 1
                        placements.append((bi, x, y, z, w, h, d))
                        placed = True
                        break
                    else:
                        # Box doesn't fit in container at all? Shouldn't happen if valid input
                        # Place at origin as fallback (may be invalid but avoids crash)
                        x, y, z = 0, 0, 0
                        new_bin = [(x, y, z, w, h, d)]
                        bins.append(new_bin)
                        bi = len(bins) - 1
                        placements.append((bi, x, y, z, w, h, d))
                        placed = True
                        break
            
            if not placed:
                # Absolute fallback
                x, y, z = 0, 0, 0
                new_bin = [(x, y, z, w, h, d)]
                bins.append(new_bin)
                bi = len(bins) - 1
                placements.append((bi, x, y, z, w, h, d))
        
        # Calculate score for this placement
        if len(placements) == n:
            bin_labels = [p[0] for p in placements]
            max_bin = max(bin_labels) if bin_labels else 0
            num_bins = max_bin + 1
            
            # Last bin occupied volume
            last_bin_vol = 0
            for p in placements:
                if p[0] == max_bin:
                    last_bin_vol += p[4] * p[5] * p[6]
            
            last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
            score = num_bins + 0.1 * (1 - last_bin_frac)
            
            if score < best_score:
                best_score = score
                best_placements = placements
    
    # If no valid placement found, use the first strategy's result
    if best_placements is None:
        # Fallback: simple shelf packing
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
