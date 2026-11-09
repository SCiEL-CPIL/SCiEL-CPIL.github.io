"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import product

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Sort boxes by volume descending, but keep track of original index
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
    
    # Try multiple packing strategies and pick best score
    def evaluate(placements):
        """Compute score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return float('inf')
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return float('inf')
        frac = last_vol / container_vol
        score = B + 0.1 * (1 - frac)
        return score
    
    def shelf_pack(box_list):
        """Simple shelf heuristic."""
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for idx, (w, h, d) in box_list:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    def max_cubes_pack(box_list):
        """
        Best-fit decreasing with candidate position generation.
        For each box, generate candidate positions from existing box corners.
        """
        # Each bin maintains a list of placed boxes: (x, y, z, w, h, d)
        bins = []  # list of lists
        
        def candidate_positions(bin_boxes, w, h, d):
            """Generate candidate (x,y,z) positions for a box of size (w,h,d) in a bin."""
            cands = [(0, 0, 0)]
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                # Corner to the right
                cands.append((bx + bw, by, bz))
                # Corner above
                cands.append((bx, by + bh, bz))
                # Corner in front
                cands.append((bx, by, bz + bd))
                # Combinations
                cands.append((bx + bw, by + bh, bz))
                cands.append((bx + bw, by, bz + bd))
                cands.append((bx, by + bh, bz + bd))
                cands.append((bx + bw, by + bh, bz + bd))
            
            seen = set()
            valid_cands = []
            for (cx, cy, cz) in cands:
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                key = (cx, cy, cz)
                if key in seen:
                    continue
                seen.add(key)
                valid_cands.append(key)
            
            # If no candidates, try (0,0,0)
            if not valid_cands:
                if w <= W and h <= H and d <= D:
                    valid_cands = [(0, 0, 0)]
                else:
                    valid_cands = []
            
            return valid_cands
        
        def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
            return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                        y1 + h1 <= y2 or y2 + h2 <= y1 or
                        z1 + d1 <= z2 or z2 + d2 <= z1)
        
        def can_place(bin_boxes, x, y, z, w, h, d):
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if is_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    return False
            return True
        
        # Greedy: for each box in sorted order, try to place in existing bins
        # Prefer bins with more existing volume (to concentrate), then smallest bin count
        placements_by_idx = {}
        
        for idx, (w, h, d) in box_list:
            placed = False
            # Try existing bins first, prefer bin with most used volume
            if bins:
                # Sort bins by volume used descending
                bin_vols = []
                for bi, bb in enumerate(bins):
                    vol = sum(bw*bh*bd for (bx, by, bz, bw, bh, bd) in bb)
                    bin_vols.append((vol, bi))
                bin_vols.sort(reverse=True)
                
                for _, bi in bin_vols:
                    bb = bins[bi]
                    cands = candidate_positions(bb, w, h, d)
                    # Sort candidates by (x+y+z) to prefer lower-left-back
                    cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
                    for (cx, cy, cz) in cands:
                        if can_place(bb, cx, cy, cz, w, h, d):
                            bb.append((cx, cy, cz, w, h, d))
                            placements_by_idx[idx] = (bi, cx, cy, cz, w, h, d)
                            placed = True
                            break
                    if placed:
                        break
            
            if not placed:
                # Open new bin
                bi = len(bins)
                bins.append([])
                bins[bi].append((0, 0, 0, w, h, d))
                placements_by_idx[idx] = (bi, 0, 0, 0, w, h, d)
        
        # Convert to original order
        result = []
        for idx in range(n):
            if idx in placements_by_idx:
                b, x, y, z, w, h, d = placements_by_idx[idx]
                # Get original box dimensions
                ow, oh, od = boxes[idx]
                result.append((b, x, y, z, ow, oh, od))
            else:
                # Shouldn't happen, but fallback
                result.append((0, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2]))
        return result
    
    # Generate placements using both methods
    best_placements = None
    best_score = float('inf')
    
    # Method 1: Shelf
    shelf_pl = shelf_pack(indexed_boxes)
    # Convert back to original order
    shelf_final = [None] * n
    for i, (idx, (w, h, d)) in enumerate(indexed_boxes):
        b, x, y, z, bw, bh, bd = shelf_pl[i]
        shelf_final[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    shelf_final = tuple(shelf_final)
    s1 = evaluate(shelf_final)
    if s1 < best_score:
        best_score = s1
        best_placements = shelf_final
    
    # Method 2: Max-cubes / best-fit
    if time.time() < deadline:
        mc_pl = max_cubes_pack(indexed_boxes)
        s2 = evaluate(mc_pl)
        if s2 < best_score:
            best_score = s2
            best_placements = tuple(mc_pl)
    
    # Method 3: Shelf with different sort orders
    if time.time() < deadline:
        # Sort by max dimension descending
        indexed_by_maxdim = sorted(enumerate(boxes), key=lambda t: -(max(t[1])))
        shelf_pl2 = shelf_pack(indexed_by_maxdim)
        shelf_final2 = [None] * n
        for i, (idx, (w, h, d)) in enumerate(indexed_by_maxdim):
            b, x, y, z, bw, bh, bd = shelf_pl2[i]
            shelf_final2[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        shelf_final2 = tuple(shelf_final2)
        s3 = evaluate(shelf_final2)
        if s3 < best_score:
            best_score = s3
            best_placements = shelf_final2
    
    # Method 4: First-fit decreasing with max-cubes
    if time.time() < deadline:
        indexed_by_vol = sorted(enumerate(boxes), key=lambda t: -(t[1][0]*t[1][1]*t[1][2]))
        # Same as max_cubes but try all bins before opening new
        # Already done in max_cubes_pack, but let's try a variant
        # Sort by (w+h+d) descending
        indexed_by_sum = sorted(enumerate(boxes), key=lambda t: -(t[1][0]+t[1][1]+t[1][2]))
        mc_pl2 = max_cubes_pack(indexed_by_sum)
        s4 = evaluate(mc_pl2)
        if s4 < best_score:
            best_score = s4
            best_placements = tuple(mc_pl2)
    
    if best_placements is None:
        # Fallback to shelf
        shelf_pl = shelf_pack(indexed_boxes)
        shelf_final = [None] * n
        for i, (idx, (w, h, d)) in enumerate(indexed_boxes):
            b, x, y, z, bw, bh, bd = shelf_pl[i]
            shelf_final[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        best_placements = tuple(shelf_final)
    
    return list(best_placements)
# EVOLVE-BLOCK-END
