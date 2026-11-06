import time
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Each bin has a list of placed boxes (x, y, z, w, h, d)
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap: overlap if not separated in any axis
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def get_corner_points(bin_boxes):
        """Generate candidate corner points from placed boxes and container origin."""
        points = set()
        points.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Points on the "outer faces" extending from each box
            if px + pw <= W:
                points.add((px + pw, py, pz))
                points.add((px + pw, py, pz + pd))
                points.add((px + pw, py + ph, pz))
                points.add((px + pw, py + ph, pz + pd))
            if py + ph <= H:
                points.add((px, py + ph, pz))
                points.add((px, py + ph, pz + pd))
            if pz + pd <= D:
                points.add((px, py, pz + pd))
            # Also add the box's own corner
            points.add((px, py, pz))
        # Filter valid points
        valid = []
        for (cx, cy, cz) in points:
            if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                valid.append((cx, cy, cz))
        return valid
    
    def place_in_bin(bin_boxes, w, h, d):
        """Try to place a box in a bin using corner-point heuristic.
        Returns (x, y, z) if placed, None otherwise."""
        corners = get_corner_points(bin_boxes)
        best_score = float('inf')
        best_pos = None
        for (cx, cy, cz) in corners:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                # Score: minimize sum of distances to origin (compactness)
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        if best_pos is None:
            # Try a simple grid scan as fallback
            for z in range(0, D - d + 1):
                for y in range(0, H - h + 1):
                    for x in range(0, W - w + 1):
                        if can_place(bin_boxes, x, y, z, w, h, d):
                            return (x, y, z)
            return None
        return best_pos
    
    def bin_volume(bin_boxes):
        return sum(w*h*d for (_, _, _, w, h, d) in bin_boxes)
    
    def score_solution(assignments):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        # assignments: list of (bin_id, box_index)
        bin_ids = set(a[0] for a in assignments)
        B = len(bin_ids)
        last_bin = max(bin_ids)
        last_bin_vol = 0
        for (bid, box_idx) in assignments:
            if bid == last_bin:
                w, h, d = boxes[box_idx]
                last_bin_vol += w * h * d
        if container_vol > 0:
            frac = last_bin_vol / container_vol
        else:
            frac = 1.0
        return B + 0.1 * (1 - frac)
    
    # Greedy assignment with corner-point heuristic
    assignments = []  # list of (bin_id, box_index)
    box_to_bin = {}
    
    for (orig_idx, w, h, d) in indexed_boxes:
        best_bin = None
        best_pos = None
        best_score = float('inf')
        
        for b in range(len(bins)):
            pos = place_in_bin(bins[b], w, h, d)
            if pos is not None:
                # Score for this placement: prefer bins that lead to compactness
                # Use the corner score plus a penalty for opening new fragmentation
                cx, cy, cz = pos
                placement_score = cx + cy + cz
                # Also consider how full the bin is (prefer fuller bins to minimize B)
                current_vol = bin_volume(bins[b])
                fullness = current_vol / container_vol if container_vol > 0 else 0
                # Lower fullness means more empty space, which is worse for bin count
                # But we want compact placement, so combine
                combined = placement_score + 50 * (1 - fullness)
                if combined < best_score:
                    best_score = combined
                    best_bin = b
                    best_pos = pos
        
        if best_bin is None:
            # Open a new bin
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        box_to_bin[orig_idx] = best_bin
        assignments.append((best_bin, orig_idx))
        
        # Check time
        if time.time() > deadline:
            break
    
    # Build final placements
    placements = []
    for (orig_idx, w, h, d) in indexed_boxes:
        if orig_idx not in box_to_bin:
            # Didn't get placed due to timeout; place in a new bin at origin
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            box_to_bin[orig_idx] = b
        b = box_to_bin[orig_idx]
        # Find the position in the bin
        for (px, py, pz, pw, ph, pd) in bins[b]:
            if pw == w and ph == h and pd == d:
                placements.append((b, px, py, pz, w, h, d))
                break
        else:
            # Should not happen
            placements.append((b, 0, 0, 0, w, h, d))
    
    # Ensure we have exactly n placements in the original order
    # The problem says "one placement per box, preserving its original ordered dimensions"
    # So we need to return placements in the order of the input boxes
    result = [None] * n
    for i, (b, px, py, pz, w, h, d) in enumerate(placements):
        # The i-th element in placements corresponds to indexed_boxes[i]
        orig_idx = indexed_boxes[i][0]
        result[orig_idx] = (b, px, py, pz, w, h, d)
    
    # Fill any None entries (shouldn't happen)
    for i in range(n):
        if result[i] is None:
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return result
