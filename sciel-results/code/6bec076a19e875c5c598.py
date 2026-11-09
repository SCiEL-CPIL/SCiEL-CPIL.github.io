"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    3D Bin Packing using First-Fit Decreasing with Corner Heuristic.
    
    Approach:
    1. Sort boxes by volume descending.
    2. For each box, try to place it in existing bins.
    3. For each bin, generate candidate positions from 'free corners' (corners of existing boxes and container origin).
    4. Filter corners that are valid (within container, no overlap).
    5. Select the best corner (lowest z, then x, then y) to maximize stacking.
    6. If no existing bin works, open a new bin.
    7. Post-processing: Local search to move boxes from the last bin to previous bins.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending, tie-break by max dimension
    sorted_indices = sorted(range(n), key=lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                                      -max(boxes[i][0], boxes[i][1], boxes[i][2])))
    
    # bins[i] is a list of placed boxes: [(x, y, z, w, h, d), ...]
    bins = []
    # placements[i] = (bin_idx, x, y, z, w, h, d)
    placements = [None] * n
    
    def overlaps(b1, b2):
        """Check if two boxes (x,y,z,w,h,d) overlap."""
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
        
    def is_valid_position(pos, box, bin_boxes):
        """Check if a position is valid for a box in a bin."""
        x, y, z = pos
        w, h, d = box
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap
        new_box = (x, y, z, w, h, d)
        for b in bin_boxes:
            if overlaps(new_box, b):
                return False
        return True

    def get_candidate_corners(bin_boxes):
        """
        Generate candidate corners for a new box.
        Candidates are:
        1. (0,0,0)
        2. (x+w, y, z) for each box
        3. (x, y+h, z) for each box
        4. (x, y, z+d) for each box
        5. (x+w, y+h, z)
        6. (x+w, y, z+d)
        7. (x, y+h, z+d)
        8. (x+w, y+h, z+d)
        We filter duplicates and invalid ones.
        """
        corners = set()
        corners.add((0, 0, 0))
        for (x, y, z, w, h, d) in bin_boxes:
            # 8 corners of the box
            corners.add((x + w, y, z))
            corners.add((x, y + h, z))
            corners.add((x, y, z + d))
            corners.add((x + w, y + h, z))
            corners.add((x + w, y, z + d))
            corners.add((x, y + h, z + d))
            corners.add((x + w, y + h, z + d))
            # Also consider corners shifted by box dimensions? No, the 8 corners of existing boxes are sufficient 
            # to generate all "supported" positions in a skyline/orthogonal packing sense if we check validity.
            # Actually, we should also consider positions where the box sits on top of another, 
            # i.e., z = z_other + d_other. This is covered by (x, y, z_other + d_other) if x,y align.
            # The set of 8 corners of all boxes plus (0,0,0) is a standard heuristic for 3D packing.
            
        # Filter corners that are within container bounds
        valid_corners = []
        for c in corners:
            if 0 <= c[0] < W and 0 <= c[1] < H and 0 <= c[2] < D:
                valid_corners.append(c)
        return valid_corners

    def find_best_position(box, bin_boxes):
        """
        Find the best position for a box in a given bin.
        Returns (x, y, z) or None.
        """
        w, h, d = box
        corners = get_candidate_corners(bin_boxes)
        
        best_pos = None
        best_score = float('inf')
        
        for c in corners:
            x, y, z = c
            if is_valid_position((x, y, z), box, bin_boxes):
                # Heuristic: Prefer lower z (bottom), then lower x, then lower y
                # This encourages stacking and filling the bottom-left-front
                score = z * 10000 + x * 100 + y
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
                    
        return best_pos

    # Phase 1: Initial Packing
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins
        # We can try to optimize bin selection: prefer bins that are "fuller" or have more space?
        # Standard FFD tries bins in order. Let's try to sort bins by how "good" they are for this box?
        # For simplicity and speed, we iterate through bins.
        # To improve consolidation, we can try bins in reverse order of creation (latest first) or 
        # by remaining volume. Let's stick to standard FFD order (0 to N-1) for stability, 
        # but maybe check the "active" bin first?
        
        for bin_idx in range(len(bins)):
            pos = find_best_position((w, h, d), bins[bin_idx])
            if pos is not None:
                x, y, z = pos
                placements[idx] = (bin_idx, x, y, z, w, h, d)
                bins[bin_idx].append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Open new bin
            bin_idx = len(bins)
            bins.append([])
            # Place at (0,0,0)
            placements[idx] = (bin_idx, 0, 0, 0, w, h, d)
            bins[bin_idx].append((0, 0, 0, w, h, d))
            
    # Phase 2: Local Search / Improvement
    # Goal: Reduce B (number of bins) and increase last-bin volume fraction.
    # Strategy: Try to move boxes from the last bin to earlier bins.
    
    def get_bin_count():
        if not placements:
            return 0
        return max(p[0] for p in placements) + 1

    def get_last_bin_volume_fraction():
        if not placements:
            return 0.0
        last_bin = max(p[0] for p in placements)
        last_bin_vol = sum(w*h*d for (_, _, _, _, w, h, d) in placements if _ == last_bin)
        container_vol = W * H * D
        return last_bin_vol / container_vol if container_vol > 0 else 0.0

    def compute_score():
        B = get_bin_count()
        last_frac = get_last_bin_volume_fraction()
        return B + 0.1 * (1 - last_frac)

    # We perform a simple local search:
    # Iterate through bins from last to first.
    # For each box in a bin, try to move it to any previous bin.
    # If it fits, move it. This might reduce the number of bins if the last bin becomes empty,
    # or it might just improve the fill of the last bin if we are moving boxes INTO the last bin?
    # No, moving boxes OUT of the last bin reduces its volume fraction, which is bad for the score term 0.1*(1-frac).
    # Wait, the score is B + 0.1*(1 - last_frac).
    # If we move a box OUT of the last bin, B might decrease (if bin becomes empty) OR stay same.
    # If B decreases, that's a big win (1.0).
    # If B stays same, last_frac decreases, so (1-frac) increases, score increases (bad).
    # So we should ONLY move boxes out of the last bin if it results in the bin becoming empty (reducing B).
    # What about moving boxes INTO the last bin from earlier bins?
    # That would increase last_frac, decreasing the penalty term. But it might increase B? No, B is max label.
    # If we move a box from bin 0 to bin 5 (last), bin 0 might become empty? If bin 0 was the only one, B drops.
    # Generally, consolidating boxes into fewer bins is the main goal.
    
    # Let's try a "compaction" heuristic:
    # 1. Try to remove the last bin by moving its boxes to previous bins.
    # 2. If successful, B decreases.
    
    improved = True
    max_iter = 10
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        # Get current bin count
        B = get_bin_count()
        if B <= 1:
            break
            
        last_bin_idx = B - 1
        # Get boxes in the last bin
        boxes_in_last = [i for i, p in enumerate(placements) if p is not None and p[0] == last_bin_idx]
        
        if not boxes_in_last:
            break
            
        # Try to move each box in the last bin to any other bin
        # We should try to move the largest ones first? Or just iterate.
        # To ensure we don't make a mess, we check if the move is valid.
        
        # We need to be careful: if we move a box, the "free space" in the source bin changes.
        # Since we are using a "check all boxes" validity method, we can just simulate the move.
        
        moved_any = False
        for box_idx in boxes_in_last:
            w, h, d = boxes[box_idx]
            
            # Try to place in bins 0 to last_bin_idx - 1
            for target_bin in range(last_bin_idx):
                # Check if box fits in target_bin
                # We need to check against all boxes in target_bin
                if is_valid_position((0,0,0), (w,h,d), bins[target_bin]): # Just a dummy check to get corners? No, we need to find a pos.
                    pos = find_best_position((w, h, d), bins[target_bin])
                    if pos is not None:
                        # It fits! Let's move it.
                        x, y, z = pos
                        
                        # Remove from last bin
                        # Find the box in bins[last_bin_idx] and remove it
                        for j, b in enumerate(bins[last_bin_idx]):
                            if b[3] == w and b[4] == h and b[5] == d and b[0] == placements[box_idx][1] and b[1] == placements[box_idx][2] and b[2] == placements[box_idx][3]:
                                bins[last_bin_idx].pop(j)
                                break
                        placements[box_idx] = (target_bin, x, y, z, w, h, d)
                        bins[target_bin].append((x, y, z, w, h, d))
                        
                        improved = True
                        moved_any = True
                        break # Move to next box in last bin
            if moved_any:
                break # Restart the loop for the new last bin? Or continue?
                # If we moved one box, the last bin might still have boxes.
                # But the structure changed. Let's break and restart the while loop to recalculate B.
                
    return placements
# EVOLVE-BLOCK-END
