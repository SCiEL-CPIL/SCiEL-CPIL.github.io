"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with aggressive consolidation.
    
    Approach:
    1. Sort boxes by volume descending.
    2. For each box, evaluate ALL existing bins and find the best (bin, position) pair.
    3. Best position minimizes a composite score favoring compactness.
    4. If no bin works, open a new bin.
    5. Aggressive local search: repeatedly try to eliminate the last bin by moving its boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending, tie-break by max dimension
    sorted_indices = sorted(range(n), key=lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                                      -max(boxes[i][0], boxes[i][1], boxes[i][2])))
    
    bins = []
    placements = [None] * n
    
    def overlaps(b1, b2):
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
        
    def find_valid_positions(box, bin_boxes):
        """Find all valid corner positions for box in bin_boxes."""
        w, h, d = box
        candidates = set()
        candidates.add((0, 0, 0))
        for (x, y, z, bw, bh, bd) in bin_boxes:
            candidates.add((x + bw, y, z))
            candidates.add((x, y + bh, z))
            candidates.add((x, y, z + bd))
            candidates.add((x + bw, y + bh, z))
            candidates.add((x + bw, y, z + bd))
            candidates.add((x, y + bh, z + bd))
            candidates.add((x + bw, y + bh, z + bd))
        
        valid = []
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            new_box = (x, y, z, w, h, d)
            if any(overlaps(new_box, b) for b in bin_boxes):
                continue
            valid.append((x, y, z))
        return valid
    
    def position_score(pos, box, bin_boxes):
        """Score a position: lower is better. Favors compact, low positions."""
        x, y, z = pos
        w, h, d = box
        # Primary: minimize z (stacking)
        # Secondary: minimize x, then y
        # Tertiary: prefer positions adjacent to existing boxes (compactness)
        base_score = z * 10000 + x * 100 + y
        
        # Bonus: prefer positions that touch existing boxes
        touch_bonus = 0
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check if new box touches this box
            if (abs(x - (bx + bw)) < 1e-9 or abs((x + w) - bx) < 1e-9 or
                abs(y - (by + bh)) < 1e-9 or abs((y + h) - by) < 1e-9 or
                abs(z - (bz + bd)) < 1e-9 or abs((z + d) - bz) < 1e-9):
                # Check if they actually share a face (not just corner)
                if (x + w > bx and bx + bw > x and y + h > by and by + bh > y and z + d > bz and bz + bd > z):
                    touch_bonus += 1
        
        return base_score - touch_bonus * 10
    
    # Phase 1: Best-Fit Decreasing
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for bin_idx in range(len(bins)):
            positions = find_valid_positions((w, h, d), bins[bin_idx])
            for pos in positions:
                s = position_score(pos, (w, h, d), bins[bin_idx])
                if s < best_score:
                    best_score = s
                    best_bin = bin_idx
                    best_pos = pos
        
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        placements[idx] = (best_bin, x, y, z, w, h, d)
        bins[best_bin].append((x, y, z, w, h, d))
    
    # Phase 2: Aggressive consolidation - try to reduce bin count
    def get_num_bins():
        if not placements:
            return 0
        return max(p[0] for p in placements) + 1
    
    def try_move_box(box_idx, from_bin, to_bin):
        """Try to move box from from_bin to to_bin. Returns True if successful."""
        w, h, d = boxes[box_idx]
        positions = find_valid_positions((w, h, d), bins[to_bin])
        if not positions:
            return False
        
        # Pick best position
        best_pos = min(positions, key=lambda p: position_score(p, (w, h, d), bins[to_bin]))
        x, y, z = best_pos
        
        # Remove from source bin
        src_box = None
        for j, b in enumerate(bins[from_bin]):
            if (b[3] == w and b[4] == h and b[5] == d and 
                b[0] == placements[box_idx][1] and b[1] == placements[box_idx][2] and b[2] == placements[box_idx][3]):
                src_box = bins[from_bin].pop(j)
                break
        
        if src_box is None:
            return False
        
        bins[to_bin].append((x, y, z, w, h, d))
        placements[box_idx] = (to_bin, x, y, z, w, h, d)
        return True
    
    # Repeatedly try to eliminate the last bin
    for _ in range(50):
        B = get_num_bins()
        if B <= 1:
            break
        
        last_bin = B - 1
        boxes_in_last = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin]
        
        if not boxes_in_last:
            break
        
        moved = False
        # Try moving boxes from last bin to other bins, largest first
        boxes_in_last.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
        
        for box_idx in boxes_in_last:
            for target in range(last_bin):
                if try_move_box(box_idx, last_bin, target):
                    moved = True
                    break
            if moved:
                break
        
        if not moved:
            break
    
    return placements
# EVOLVE-BLOCK-END
