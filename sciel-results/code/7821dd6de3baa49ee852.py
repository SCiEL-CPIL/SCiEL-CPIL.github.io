"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from copy import deepcopy

def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with proper free-cuboid management.
    
    Improvements over previous version:
    1. Proper splitting of free cuboids into non-overlapping sub-cuboids
    2. Better scoring that balances waste and position (bottom-left-front)
    3. Local search improvement pass to reduce bin count and fill last bin better
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending, with tie-breaking by max dimension
    sorted_indices = sorted(range(n), key=lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                                      -max(boxes[i][0], boxes[i][1], boxes[i][2])))
    
    # bins[i] is a list of free cuboids: (x, y, z, w, h, d)
    bins = []
    # placements[i] = (bin_idx, x, y, z, w, h, d)
    placements = [None] * n
    
    def add_bin():
        bins.append([(0, 0, 0, W, H, D)])
        return len(bins) - 1
    
    def split_cuboid(fx, fy, fz, fw, fh, fd, x, y, z, w, h, d):
        """
        Split a free cuboid (fx, fy, fz, fw, fh, fd) by placing a box at (x, y, z) 
        with dimensions (w, h, d). Returns a list of new non-overlapping free cuboids.
        """
        new_cuboids = []
        
        # Split along x-axis: left and right
        if x > fx:
            new_cuboids.append((fx, fy, fz, x - fx, fh, fd))
        if x + w < fx + fw:
            new_cuboids.append((x + w, fy, fz, fx + fw - (x + w), fh, fd))
        
        # For the region where x is within [x, x+w], split along y and z
        # Middle region: x in [x, x+w]
        if x < fx + fw and x + w > fx:
            mid_x = max(x, fx)
            mid_w = min(x + w, fx + fw) - mid_x
            if mid_w > 0:
                # Split along y-axis within this x-range
                if y > fy:
                    new_cuboids.append((mid_x, fy, fz, mid_w, y - fy, fd))
                if y + h < fy + fh:
                    new_cuboids.append((mid_x, y + h, fz, mid_w, fy + fh - (y + h), fd))
                
                # For the region where y is within [y, y+h], split along z
                if y < fy + fh and y + h > fy:
                    mid_y = max(y, fy)
                    mid_h = min(y + h, fy + fh) - mid_y
                    if mid_h > 0:
                        if z > fz:
                            new_cuboids.append((mid_x, mid_y, fz, mid_w, mid_h, z - fz))
                        if z + d < fz + fd:
                            new_cuboids.append((mid_x, mid_y, z + d, mid_w, mid_h, fz + fd - (z + d)))
        
        # Filter out any cuboids with zero or negative dimensions
        valid_cuboids = []
        for cub in new_cuboids:
            cx, cy, cz, cw, ch, cd = cub
            if cw > 0 and ch > 0 and cd > 0:
                valid_cuboids.append(cub)
        
        return valid_cuboids
    
    def try_place_in_bin(bin_idx, w, h, d):
        """
        Try to place a box of size (w, h, d) in bin bin_idx.
        Returns (x, y, z) if placed, else None.
        Also updates the free cuboids in bins[bin_idx].
        """
        free_cuboids = bins[bin_idx]
        best_pos = None
        best_score = float('inf')
        best_new_cuboids = None
        best_idx = -1
        
        for i, (fx, fy, fz, fw, fh, fd) in enumerate(free_cuboids):
            if w <= fw and h <= fh and d <= fd:
                # Try placing at bottom-left-front of this cuboid
                x, y, z = fx, fy, fz
                new_cuboids = split_cuboid(fx, fy, fz, fw, fh, fd, x, y, z, w, h, d)
                
                # Calculate total volume of new free cuboids (waste)
                waste = sum(cw * ch * cd for _, _, _, cw, ch, cd in new_cuboids)
                
                # Position score: prefer bottom (z), then left (x), then front (y)
                # Use a weighted sum that heavily favors lower z, then lower x, then lower y
                pos_score = z * 1000000 + x * 1000 + y
                
                # Combined score: minimize waste, then position
                total_score = waste + pos_score * 0.001
                
                if total_score < best_score:
                    best_score = total_score
                    best_pos = (x, y, z)
                    best_new_cuboids = new_cuboids
                    best_idx = i
        
        if best_pos is not None:
            # Update the free cuboids: replace the used cuboid with new ones
            new_free_cuboids = []
            for j, cuboid in enumerate(free_cuboids):
                if j == best_idx:
                    new_free_cuboids.extend(best_new_cuboids)
                else:
                    new_free_cuboids.append(cuboid)
            bins[bin_idx] = new_free_cuboids
            return best_pos
        return None
    
    # Phase 1: Initial packing with Best-Fit Decreasing
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins, preferring bins that are already more filled
        # to encourage consolidation
        bin_order = sorted(range(len(bins)), 
                          key=lambda bi: -sum(cw*ch*cd for _,_,_,cw,ch,cd in bins[bi]), 
                          reverse=True)
        
        for bin_idx in bin_order:
            pos = try_place_in_bin(bin_idx, w, h, d)
            if pos is not None:
                x, y, z = pos
                placements[idx] = (bin_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            bin_idx = add_bin()
            placements[idx] = (bin_idx, 0, 0, 0, w, h, d)
            # Update the free cuboids for the new bin
            new_cuboids = split_cuboid(0, 0, 0, W, H, D, 0, 0, 0, w, h, d)
            bins[bin_idx] = new_cuboids
    
    # Phase 2: Local search improvement
    # Try to reduce the number of bins by moving boxes from higher-indexed bins to lower ones
    # Also try to fill the last bin more to improve the volume fraction score
    
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
    
    best_score = compute_score()
    best_placements = deepcopy(placements)
    
    # Try local search: for each box in a higher bin, try to move it to a lower bin
    max_iterations = 100
    iteration = 0
    improved = True
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Get unique bin indices
        bin_indices = sorted(set(p[0] for p in placements if p is not None))
        
        for bin_idx in reversed(bin_indices):  # Start from highest bin
            if len(bin_indices) <= 1:
                break
                
            # Get boxes in this bin
            boxes_in_bin = [i for i, p in enumerate(placements) if p is not None and p[0] == bin_idx]
            
            for box_idx in boxes_in_bin:
                w, h, d = boxes[box_idx]
                
                # Try to move this box to a lower bin
                for target_bin in range(bin_idx):
                    # Save current state
                    old_placements = deepcopy(placements)
                    old_bins = deepcopy(bins)
                    
                    # Remove box from current bin
                    # Rebuild free cuboids for the current bin by removing the box
                    # This is complex, so let's just try to place it in target bin
                    # and if it fits, we'll rebuild the source bin
                    
                    pos = try_place_in_bin(target_bin, w, h, d)
                    if pos is not None:
                        x, y, z = pos
                        placements[box_idx] = (target_bin, x, y, z, w, h, d)
                        
                        # Rebuild the source bin's free cuboids
                        # We need to reconstruct what the free space would be without this box
                        # This is tricky. Let's skip this and just keep the old state if it doesn't help
                        # Actually, we should rebuild properly. For now, let's compute the score
                        # and if it's better, we'll keep it. But we need valid free cuboids.
                        
                        # Let's rebuild the source bin by placing all other boxes in it
                        # This is expensive but N is small
                        other_boxes = [i for i, p in enumerate(placements) 
                                      if p is not None and p[0] == bin_idx and i != box_idx]
                        
                        # Clear the bin and re-add other boxes
                        bins[bin_idx] = [(0, 0, 0, W, H, D)]
                        for other_idx in other_boxes:
                            ow, oh, od = boxes[other_idx]
                            opos = try_place_in_bin(bin_idx, ow, oh, od)
                            if opos is None:
                                # This shouldn't happen if the original packing was valid
                                # Restore and break
                                placements = old_placements
                                bins = old_bins
                                break
                        else:
                            # Check if the move improved the score
                            new_score = compute_score()
                            if new_score < best_score - 1e-9:
                                best_score = new_score
                                best_placements = deepcopy(placements)
                                improved = True
                            else:
                                # Restore
                                placements = old_placements
                                bins = old_bins
                    else:
                        # Couldn't place in target bin, restore
                        placements = old_placements
                        bins = old_bins
    
    # Use the best placements found
    placements = best_placements
    
    return placements
# EVOLVE-BLOCK-END
