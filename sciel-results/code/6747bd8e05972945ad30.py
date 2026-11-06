"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Best Fit Decreasing with Skyline (Maximal Free Rectangles in 2D per Z-level).
    Uses a simpler and faster approach: for each box, try all existing bins and find the position
    that minimizes the resulting "waste" or maximizes tightness.
    """
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # bins: list of lists of placed boxes (x, y, z, w, h, d)
    bins = []
    placements = [None] * len(boxes)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True
    
    def get_candidate_positions(bin_boxes, w, h, d):
        """
        Generate candidate positions for placing a box of size (w,h,d) in a bin.
        Uses extreme points derived from existing boxes.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Points to the right, front, and top of each box
            if bx + bw <= W:
                candidates.add((bx + bw, by, bz))
            if by + bh <= H:
                candidates.add((bx, by + bh, bz))
            if bz + bd <= D:
                candidates.add((bx, by, bz + bd))
            
            # Also add corner combinations for better packing
            if bx + bw <= W and by + bh <= H:
                candidates.add((bx + bw, by + bh, bz))
            if bx + bw <= W and bz + bd <= D:
                candidates.add((bx + bw, by, bz + bd))
            if by + bh <= H and bz + bd <= D:
                candidates.add((bx, by + bh, bz + bd))
            if bx + bw <= W and by + bh <= H and bz + bd <= D:
                candidates.add((bx + bw, by + bh, bz + bd))
        
        # Filter valid candidates
        valid_candidates = []
        for (x, y, z) in candidates:
            if can_place(bin_boxes, x, y, z, w, h, d):
                valid_candidates.append((x, y, z))
        
        return valid_candidates
    
    def score_position(bin_boxes, x, y, z, w, h, d):
        """
        Score a position: lower is better.
        We want to minimize the "top" (z + d) to keep the bin short,
        then minimize x, then y to keep it compact.
        Also consider how well it fills the space (less waste is better).
        """
        # Primary: minimize the new height (z + d)
        # Secondary: minimize x, then y
        return (z + d, x, y)
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        best_bin_index = -1
        best_pos = None
        best_score = None
        
        for b_idx in range(len(bins)):
            bin_boxes = bins[b_idx]
            candidates = get_candidate_positions(bin_boxes, w, h, d)
            
            for (x, y, z) in candidates:
                sc = score_position(bin_boxes, x, y, z, w, h, d)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_bin_index = b_idx
                    best_pos = (x, y, z)
        
        if best_bin_index != -1:
            x, y, z = best_pos
            bins[best_bin_index].append((x, y, z, w, h, d))
            placements[idx] = (best_bin_index, x, y, z, w, h, d)
        else:
            # Open new bin
            new_bin_index = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[idx] = (new_bin_index, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
