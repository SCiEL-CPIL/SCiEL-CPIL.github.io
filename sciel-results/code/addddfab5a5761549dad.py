"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Define sorting strategies
    def sort_by_volume(b):
        return sorted(b, key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def sort_by_max_dim(b):
        return sorted(b, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        
    def sort_by_surface_area(b):
        return sorted(b, key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)

    strategies = [sort_by_volume, sort_by_max_dim, sort_by_surface_area]
    
    def run_packing(sorted_boxes):
        """
        sorted_boxes: list of (orig_idx, w, h, d)
        Returns: (placements, num_bins, last_bin_fill_ratio)
        """
        bins = [] # List of lists of (w, h, d, x, y, z)
        
        def can_place(bin_boxes, w, h, d, x, y, z):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                # Check overlap
                if (x < bx + bw and x + w > bx and
                    y < by + bh and y + h > by and
                    z < bz + bd and z + d > bz):
                    return False
            return True
            
        def find_best_position(bin_boxes, w, h, d):
            # Generate candidates
            candidates = [(0, 0, 0)]
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                # Right
                candidates.append((bx + bw, by, bz))
                # Back
                candidates.append((bx, by + bh, bz))
                # Top
                candidates.append((bx, by, bz + bd))
                
            # Remove duplicates and invalid bounds quickly
            valid_candidates = set()
            for c in candidates:
                if 0 <= c[0] <= W-w and 0 <= c[1] <= H-h and 0 <= c[2] <= D-d:
                    valid_candidates.add(c)
                    
            best_pos = None
            best_key = None
            for (x, y, z) in valid_candidates:
                if can_place(bin_boxes, w, h, d, x, y, z):
                    # Heuristic: Minimize Z, then Y, then X to pack towards origin
                    key = (z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (x, y, z)
                        
            return best_pos

        placements = [None] * n
        
        for orig_idx, w, h, d in sorted_boxes:
            placed = False
            # Try to place in existing bins
            for b_idx, bin_boxes in enumerate(bins):
                pos = find_best_position(bin_boxes, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bin_boxes.append((w, h, d, x, y, z))
                    placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
                    
            if not placed:
                # Start new bin
                bins.append([(w, h, d, 0, 0, 0)])
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        if not bins:
            return placements, 0, 0.0
            
        # Calculate score components for this run
        num_bins = len(bins)
        # Find last bin (highest index)
        last_bin_idx = max(p[0] for p in placements if p is not None)
        
        # Calculate occupied volume in last bin
        last_bin_vol = 0
        for b_idx, x, y, z, w, h, d in placements:
            if b_idx == last_bin_idx:
                last_bin_vol += w * h * d
                
        container_vol = W * H * D
        fill_ratio = last_bin_vol / container_vol if container_vol > 0 else 0.0
        
        return placements, num_bins, fill_ratio

    best_result = None
    best_score = float('inf')
    
    for strategy in strategies:
        indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        sorted_boxes = strategy(indexed_boxes)
        placements, num_bins, fill_ratio = run_packing(sorted_boxes)
        
        # Score: B + 0.1 * (1 - fill_ratio)
        # We want to minimize this.
        score = num_bins + 0.1 * (1 - fill_ratio)
        
        if score < best_score:
            best_score = score
            best_result = placements
            
    return best_result

# EVOLVE-BLOCK-END
