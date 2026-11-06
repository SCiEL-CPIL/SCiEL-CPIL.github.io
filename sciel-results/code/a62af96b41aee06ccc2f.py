"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box can be placed at position (x,y,z) in the given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check for overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_spot(bin_placements, w, h, d):
        """Find the best spot for a box in a bin using frontier points."""
        # Generate candidate positions from existing placements
        candidates = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_placements, x, y, z, w, h, d):
                    valid.append((x, y, z))
        
        if not valid:
            return None
        
        # Sort by (z, y, x) to prefer lower positions (better packing)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid[0]
    
    def pack_ffd():
        """First-Fit-Decreasing packing."""
        bins = []  # list of lists of placements
        placements = [None] * n
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
            placed = False
            for b_idx, bin_placements in enumerate(bins):
                spot = find_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    bin_placements.append((x, y, z, w, h, d))
                    placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                # Start a new bin
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        
        return placements
    
    def score_placements(placements):
        """Calculate the score for a set of placements."""
        if not placements:
            return 0.0
        
        # Find all used bins
        bin_ids = set(p[0] for p in placements if p is not None)
        B = len(bin_ids)
        
        # Find the last bin (largest label)
        last_bin = max(bin_ids)
        
        # Calculate occupied volume in the last bin
        last_bin_vol = 0
        for p in placements:
            if p is not None and p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        
        last_bin_fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - last_bin_fraction)
    
    # Try FFD first
    best_placements = pack_ffd()
    best_score = score_placements(best_placements)
    
    # Try to improve with local search if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Simple local search: try moving boxes between bins
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                if best_placements[i] is None:
                    continue
                    
                orig_idx, x, y, z, w, h, d = best_placements[i]
                
                # Try moving this box to another bin
                for target_bin in range(orig_idx + 1, orig_idx + 5):
                    if target_bin == orig_idx:
                        continue
                    
                    # Get placements in target bin
                    target_placements = []
                    for j in range(n):
                        if best_placements[j] is not None and best_placements[j][0] == target_bin:
                            px, py, pz, pw, ph, pd = best_placements[j][1:7]
                            target_placements.append((px, py, pz, pw, ph, pd))
                    
                    spot = find_spot(target_placements, w, h, d)
                    if spot is not None:
                        # Tentatively move
                        old_placement = best_placements[i]
                        best_placements[i] = (target_bin, spot[0], spot[1], spot[2], w, h, d)
                        
                        new_score = score_placements(best_placements)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_placements[i] = old_placement
                        
                        if improved:
                            break
    
    # Verify validity
    def is_valid(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        # Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True
    
    if not is_valid(best_placements):
        # Fallback to simple shelf packing
        best_placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return best_placements
# EVOLVE-BLOCK-END
