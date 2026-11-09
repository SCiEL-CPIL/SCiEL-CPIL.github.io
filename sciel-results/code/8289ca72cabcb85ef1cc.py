"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Free Space Management with Region Merging.
    
    This approach maintains a list of maximal empty rectangular regions (free spaces)
    for each bin. When a box is placed, the containing region is split into up to 
    3 new regions. After each insertion, we attempt to merge adjacent free regions 
    that form a single rectangle. This keeps the number of free regions low, 
    improving search speed and packing density by reducing fragmentation.
    
    Placement heuristic: For each box, we search existing bins for a free region
    that fits the box. We prioritize regions with the smallest (x, y, z) coordinates
    to pack from the origin. If no existing bin fits, a new bin is opened.
    """
    W, H, D = container
    
    # Each bin is a list of free regions: (x, y, z, w, h, d)
    bins = []
    
    def _robust_merge(regions):
        """
        Repeatedly merge any two adjacent regions until no more merges are possible.
        """
        regions = list(regions)
        changed = True
        while changed:
            changed = False
            n = len(regions)
            for i in range(n):
                if i >= len(regions): break
                r1 = regions[i]
                x1, y1, z1, w1, h1, d1 = r1
                for j in range(i + 1, len(regions)):
                    if j >= len(regions): break
                    r2 = regions[j]
                    x2, y2, z2, w2, h2, d2 = r2
                    
                    merged_box = None
                    
                    # Check X merge
                    if y1 == y2 and h1 == h2 and z1 == z2 and d1 == d2:
                        if x1 + w1 == x2:
                            merged_box = (x1, y1, z1, w1 + w2, h1, d1)
                        elif x2 + w2 == x1:
                            merged_box = (x2, y1, z1, w1 + w2, h1, d1)
                    
                    # Check Y merge
                    if merged_box is None and x1 == x2 and w1 == w2 and z1 == z2 and d1 == d2:
                        if y1 + h1 == y2:
                            merged_box = (x1, y1, z1, w1, h1 + h2, d1)
                        elif y2 + h2 == y1:
                            merged_box = (x1, y2, z1, w1, h1 + h2, d1)
                    
                    # Check Z merge
                    if merged_box is None and x1 == x2 and w1 == w2 and y1 == y2 and h1 == h2:
                        if z1 + d1 == z2:
                            merged_box = (x1, y1, z1, w1, h1, d1 + d2)
                        elif z2 + d2 == z1:
                            merged_box = (x1, y1, z2, w1, h1, d1 + d2)
                    
                    if merged_box is not None:
                        # Replace r1 with merged, remove r2
                        regions[i] = merged_box
                        regions.pop(j)
                        changed = True
                        # Restart the inner loop with the new region i
                        break 
                # If we broke inner loop, we should restart outer loop to check new i
                if changed:
                    break
        return regions

    def split_region(region, box):
        """
        Split a free region by placing a box at its origin.
        Returns list of new free regions.
        """
        rx, ry, rz, rw, rh, rd = region
        bw, bh, bd = box
        
        new_regions = []
        
        # Region to the right (X+)
        if rw > bw:
            new_regions.append((rx + bw, ry, rz, rw - bw, rh, rd))
        
        # Region above (Y+)
        if rh > bh:
            new_regions.append((rx, ry + bh, rz, bw, rh - bh, rd))
            
        # Region behind (Z+)
        if rd > bd:
            new_regions.append((rx, ry, rz + bd, bw, bh, rd - bd))
            
        return new_regions

    def find_and_place(bins, box):
        """
        Try to place a box in existing bins.
        """
        w, h, d = box
        
        for bi in range(len(bins)):
            free_regions = bins[bi]
            
            best_idx = -1
            best_pos = (float('inf'), float('inf'), float('inf'))
            
            for i, region in enumerate(free_regions):
                rx, ry, rz, rw, rh, rd = region
                if rw >= w and rh >= h and rd >= d:
                    pos = (rx, ry, rz)
                    if pos < best_pos:
                        best_pos = pos
                        best_idx = i
            
            if best_idx != -1:
                region = free_regions[best_idx]
                rx, ry, rz, rw, rh, rd = region
                
                # Remove the region
                free_regions.pop(best_idx)
                
                # Split and add new regions
                new_regions = split_region(region, (w, h, d))
                free_regions.extend(new_regions)
                
                # Merge regions to keep the list clean
                bins[bi] = _robust_merge(free_regions)
                
                return (bi, rx, ry, rz)
        
        return None

    placements = []
    
    for box in boxes:
        w, h, d = box
        
        result = find_and_place(bins, (w, h, d))
        
        if result is not None:
            bi, x, y, z = result
            placements.append((bi, x, y, z, w, h, d))
        else:
            # Open a new bin
            bi = len(bins)
            # Initial region is the whole container
            initial_region = (0, 0, 0, W, H, D)
            new_regions = split_region(initial_region, (w, h, d))
            bins.append(new_regions)
            
            placements.append((bi, 0, 0, 0, w, h, d))
            
    return placements
# EVOLVE-BLOCK-END
