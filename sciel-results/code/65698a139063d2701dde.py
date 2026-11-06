"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # We'll use a list of occupied boxes: (x, y, z, w, h, d, box_idx)
    occupied = []
    placements = [None] * len(boxes)
    
    def get_candidates(w, h, d):
        # Generate candidate positions: bottom-left-front corners
        # x in [0, W-w], y in [0, H-h], z in [0, D-d]
        # But we only need to check against occupied boxes.
        # Candidate x: 0 and (ox + ow) for each occupied box ox, ow
        # Similarly for y, z
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (ox, oy, oz, ow, oh, od, _) in occupied:
            if ox + ow <= W - w:
                xs.add(ox + ow)
            if ox <= W - w:
                xs.add(ox)  # Not directly useful but keep for completeness? No, we need gaps.
                # Actually, the candidate x positions are:
                # 1. 0
                # 2. ox + ow for each occupied box (if <= W-w)
                # 3. ox for each occupied box? No, that would overlap unless ox > x+w.
                # The standard "skyline" or "bottom-left" rule uses:
                # x candidates: 0, and (ox + ow) for all occupied boxes where ox + ow <= W-w
                # Wait, we also need to consider x positions that are to the right of an occupied box
                # but not necessarily touching? No, for packing, the optimal position is usually
                # touching some boundary or another box.
                # Let's refine:
                # x candidates: 0, and (ox + ow) for each occupied box.
                # Also, we might need x such that x + w = ox? i.e., x = ox - w.
                # So x candidates: 0, ox + ow, ox - w (if >= 0).
                pass
        
        # Let's use a more robust candidate generation:
        # For each dimension, the candidates are:
        # 0, and for each occupied box, (ox + ow) and (ox - w) (if valid)
        
        xs = {0}
        ys = {0}
        zs = {0}
        for (ox, oy, oz, ow, oh, od, _) in occupied:
            # x candidates
            if ox + ow <= W - w:
                xs.add(ox + ow)
            if ox - w >= 0:
                xs.add(ox - w)
            # y candidates
            if oy + oh <= H - h:
                ys.add(oy + oh)
            if oy - h >= 0:
                ys.add(oy - h)
            # z candidates
            if oz + od <= D - d:
                zs.add(oz + od)
            if oz - d >= 0:
                zs.add(oz - d)
        
        # Filter candidates to be within bounds
        xs = {x for x in xs if 0 <= x <= W - w}
        ys = {y for y in ys if 0 <= y <= H - h}
        zs = {z for z in zs if 0 <= z <= D - d}
        
        # If no candidates, add 0
        if not xs: xs = {0}
        if not ys: ys = {0}
        if not zs: zs = {0}
        
        return list(xs), list(ys), list(zs)
    
    def is_valid(x, y, z, w, h, d):
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with occupied boxes
        for (ox, oy, oz, ow, oh, od, _) in occupied:
            # Overlap if intervals intersect in all 3 dimensions
            if not (x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y or z + d <= oz or oz + od <= z):
                return False
        return True
    
    def place_box(box_idx, w, h, d):
        # Try all permutations of (w, h, d)
        for perm in permutations([w, h, d]):
            pw, ph, pd = perm
            # Get candidate positions
            xs, ys, zs = get_candidates(pw, ph, pd)
            # Try all combinations
            # To improve performance, we can sort candidates by some heuristic (e.g., smallest x, then y, then z)
            # But for now, just iterate
            for x in xs:
                for y in ys:
                    for z in zs:
                        if is_valid(x, y, z, pw, ph, pd):
                            # Place the box
                            occupied.append((x, y, z, pw, ph, pd, box_idx))
                            placements[box_idx] = (0, x, y, z, pw, ph, pd)
                            return True
        return False
    
    # Try to place each box in sorted order
    for box_idx, (w, h, d) in indexed_boxes:
        if not place_box(box_idx, w, h, d):
            # Mark as not placed
            placements[box_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Ensure all placements are set
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
