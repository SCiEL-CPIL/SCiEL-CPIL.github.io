"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sorting strategies
    def sort_by_volume(b):
        return sorted(b, key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def sort_by_max_dim(b):
        return sorted(b, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        
    def sort_by_surface_area(b):
        return sorted(b, key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)
    
    def sort_by_aspect(b):
        # Sort by volume but break ties by aspect ratio (prefer more cubic)
        def aspect_key(item):
            idx, w, h, d = item
            vol = w * h * d
            # Prefer boxes with more balanced dimensions
            max_dim = max(w, h, d)
            min_dim = min(w, h, d)
            return (-vol, max_dim / max(min_dim, 1))
        return sorted(b, key=aspect_key)

    strategies = [sort_by_volume, sort_by_max_dim, sort_by_surface_area, sort_by_aspect]
    
    def run_packing(sorted_boxes):
        """
        sorted_boxes: list of (orig_idx, w, h, d)
        Returns: (placements, num_bins, last_bin_fill_ratio)
        Uses Best Fit Decreasing (BFD) with expanded candidate generation.
        """
        bins = [] # List of lists of (w, h, d, x, y, z)
        placements = [None] * n
        
        def can_place(bin_boxes, w, h, d, x, y, z):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                if (x < bx + bw and x + w > bx and
                    y < by + bh and y + h > by and
                    z < bz + bd and z + d > bz):
                    return False
            return True
            
        def find_best_position(bin_boxes, w, h, d):
            # Generate candidates: Origin + all edge-aligned positions
            candidates = set()
            candidates.add((0, 0, 0))
            
            # Collect all unique x, y, z coordinates from existing boxes
            x_coords = {0}
            y_coords = {0}
            z_coords = {0}
            
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                x_coords.add(bx)
                x_coords.add(bx + bw)
                y_coords.add(by)
                y_coords.add(by + bh)
                z_coords.add(bz)
                z_coords.add(bz + bd)
            
            # Generate all combinations of these coordinates
            for x in x_coords:
                for y in y_coords:
                    for z in z_coords:
                        candidates.add((x, y, z))
            
            # Also add positions just past each box face
            for (bw, bh, bd, bx, by, bz) in bin_boxes:
                # Right face
                if bx + bw <= W - w:
                    for y in y_coords:
                        for z in z_coords:
                            candidates.add((bx + bw, y, z))
                # Back face
                if by + bh <= H - h:
                    for x in x_coords:
                        for z in z_coords:
                            candidates.add((x, by + bh, z))
                # Top face
                if bz + bd <= D - d:
                    for x in x_coords:
                        for y in y_coords:
                            candidates.add((x, y, bz + bd))
                
                # Corner combinations
                if bx + bw <= W - w and by + bh <= H - h:
                    for z in z_coords:
                        candidates.add((bx + bw, by + bh, z))
                if bx + bw <= W - w and bz + bd <= D - d:
                    for y in y_coords:
                        candidates.add((bx + bw, y, bz + bd))
                if by + bh <= H - h and bz + bd <= D - d:
                    for x in x_coords:
                        candidates.add((x, by + bh, bz + bd))
                if bx + bw <= W - w and by + bh <= H - h and bz + bd <= D - d:
                    candidates.add((bx + bw, by + bh, bz + bd))
            
            valid_candidates = []
            for c in candidates:
                if 0 <= c[0] <= W-w and 0 <= c[1] <= H-h and 0 <= c[2] <= D-d:
                    valid_candidates.append(c)
                    
            best_pos = None
            best_key = None
            for (x, y, z) in valid_candidates:
                if can_place(bin_boxes, w, h, d, x, y, z):
                    # Heuristic: Minimize Z (pack low), then Y, then X
                    # Also consider how well the box fits (less wasted space is better)
                    key = (z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (x, y, z)
                        
            return best_pos

        for orig_idx, w, h, d in sorted_boxes:
            best_bin_idx = -1
            best_pos = None
            best_key = None
            best_waste = float('inf')
            
            # Try to place in existing bins (Best Fit)
            for b_idx, bin_boxes in enumerate(bins):
                pos = find_best_position(bin_boxes, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    key = (z, y, x)
                    # Calculate waste: how much empty space is created
                    # Simple heuristic: prefer positions that minimize the max coordinate
                    waste = z * (W - x - w) + y * (H - y - h) + (D - z - d) * (W - x - w) * (H - y - h)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_bin_idx = b_idx
                        
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].append((w, h, d, x, y, z))
                placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                # Start new bin
                bins.append([(w, h, d, 0, 0, 0)])
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        if not bins:
            return placements, 0, 0.0
            
        num_bins = len(bins)
        last_bin_idx = max(p[0] for p in placements if p is not None)
        
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
        
        score = num_bins + 0.1 * (1 - fill_ratio)
        
        if score < best_score:
            best_score = score
            best_result = placements
            
    return best_result
# EVOLVE-BLOCK-END
