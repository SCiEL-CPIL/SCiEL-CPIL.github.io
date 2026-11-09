"""3D Bin Packing using Global Best-Fit Decreasing with Corner Candidates."""
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume in descending order
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # Each bin stores: list of placed boxes (orig_idx, x, y, z, w, h, d)
    bins = []
    
    def get_candidates(bin_boxes, w, h, d):
        """
        Generate candidate positions for a new box in a bin.
        Candidates are corners of existing boxes and the origin.
        """
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        
        # Corners of existing boxes
        for (_, px, py, pz, pw, ph, pd) in bin_boxes:
            # 8 corners of the existing box
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # The 8th corner (px, py, pz) is usually the start of the box, 
            # but we might place another box at the same spot? No, overlap.
            # However, if we place a box *next to* it, we use the other corners.
            # We don't need (px, py, pz) as a candidate for *new* box start 
            # unless it's empty, but the box occupies that space.
            # Actually, if we place a box at (px, py, pz), it overlaps.
            # So we only need the "outward" corners.
            
        return list(candidates)

    def is_valid(pos, bin_boxes, w, h, d):
        """
        Check if a position is valid:
        1. In bounds.
        2. No overlap.
        3. Supported (z=0 or supported by boxes below).
        """
        x, y, z = pos
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Overlap check
        for (_, px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        
        # Support check
        if z > 0:
            support_area = 0.0
            box_area = w * h
            for (_, px, py, pz, pw, ph, pd) in bin_boxes:
                if pz + pd == z:
                    # Check overlap in XY
                    overlap_x = max(0, min(x + w, px + pw) - max(x, px))
                    overlap_y = max(0, min(y + h, py + ph) - max(y, py))
                    support_area += overlap_x * overlap_y
            # Require some support
            if support_area < 1e-6:
                return False
        
        return True

    def evaluate_position(pos, bin_boxes, w, h, d):
        """
        Return a score tuple for sorting positions.
        Lower is better.
        Primary: Z (height)
        Secondary: Y (depth)
        Tertiary: X (width)
        """
        return (pos[2], pos[1], pos[0])

    # Process boxes
    for (orig_idx, w, h, d) in indexed_boxes:
        best_bin_idx = -1
        best_pos = None
        best_score = None
        
        # Check existing bins
        for bin_idx in range(len(bins)):
            bin_boxes = bins[bin_idx]
            candidates = get_candidates(bin_boxes, w, h, d)
            
            for pos in candidates:
                if is_valid(pos, bin_boxes, w, h, d):
                    score = evaluate_position(pos, bin_boxes, w, h, d)
                    # We want the global best position across all bins.
                    # Tie-breaker: Prefer bin with more boxes? Or just first found?
                    # Let's add bin index as a final tie-breaker to prefer earlier bins
                    full_score = (score[0], score[1], score[2], bin_idx)
                    if best_score is None or full_score < best_score:
                        best_score = full_score
                        best_bin_idx = bin_idx
                        best_pos = pos
        
        if best_bin_idx != -1:
            # Place in existing bin
            x, y, z = best_pos
            bins[best_bin_idx].append((orig_idx, x, y, z, w, h, d))
        else:
            # Create new bin
            # Place at origin
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
            
    # Build result in original order
    placements = [None] * n
    for bin_idx, bin_boxes in enumerate(bins):
        for (orig_idx, x, y, z, w, h, d) in bin_boxes:
            placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
            
    return placements
