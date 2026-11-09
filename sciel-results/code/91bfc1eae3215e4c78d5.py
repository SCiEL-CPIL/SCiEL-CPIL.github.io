"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Binary Space Partitioning (BSP) trees.
    
    Each bin maintains a BSP tree that recursively subdivides the container
    into empty rectangular subspaces. When placing a box, we search for the
    first empty rectangular region (in a defined order) that can contain the
    box, then insert it by splitting that region.
    
    We process boxes in the order given. For each box, we try to place it in
    existing bins (starting from bin 0) by searching their BSP trees for a
    suitable empty space. If no existing bin can accommodate the box, we open
    a new bin.
    
    The search order within a BSP tree is: traverse nodes in a way that prefers
    lower x, then lower y, then lower z positions (to pack densely from the
    origin).
    
    Return: list of (bin, x, y, z, w, h, d) for each box in input order.
    """
    W, H, D = container
    container_vol = W * H * D if W > 0 and H > 0 and D > 0 else 1
    
    # BSP Node: represents an empty rectangular region
    # A node is either:
    #   - A leaf: empty space defined by (x, y, z, w, h, d)
    #   - An internal node: split into children
    # For simplicity, we'll use a list-based approach where each "region" is
    # a tuple (x, y, z, w, h, d) and we maintain a list of free regions per bin.
    # When we place a box in a region, we split that region into up to 3 new regions.
    
    # Each bin: list of free rectangular regions (x, y, z, w, h, d)
    bins = []  # list of lists of regions
    
    def split_region(region, box):
        """
        Given a free region and a box to place at the corner (region's x,y,z),
        split the region into up to 3 new free regions around the box.
        The box is placed at (region[0], region[1], region[2]).
        Returns list of new free regions.
        """
        rx, ry, rz, rw, rh, rd = region
        bw, bh, bd = box
        
        # Box is placed at (rx, ry, rz) with dimensions (bw, bh, bd)
        # Remaining space:
        # 1. To the right of the box (x: rx+bw to rx+rw)
        # 2. Above the box (y: ry+bh to ry+rh) but only in the x-range of the box
        # 3. Behind the box (z: rz+bd to rz+rd) but only in the x,y range of the box
        
        new_regions = []
        
        # Region to the right: (rx+bw, ry, rz, rw-bw, rh, rd)
        if rw - bw > 0:
            new_regions.append((rx + bw, ry, rz, rw - bw, rh, rd))
        
        # Region above: (rx, ry+bh, rz, bw, rh-bh, rd)
        if rh - bh > 0:
            new_regions.append((rx, ry + bh, rz, bw, rh - bh, rd))
        
        # Region behind: (rx, ry, rz+bd, bw, bh, rd-bd)
        if rd - bd > 0:
            new_regions.append((rx, ry, rz + bd, bw, bh, rd - bd))
        
        return new_regions
    
    def find_and_place(bins, box):
        """
        Try to place box (w, h, d) in one of the existing bins.
        Returns (bin_index, x, y, z) if successful, or None.
        """
        w, h, d = box
        
        for bi in range(len(bins)):
            free_regions = bins[bi]
            # Sort free regions by (x, y, z) to prefer lower positions
            # We'll iterate through regions and find the first one that fits
            # To be more efficient, we could sort, but let's just scan
            # Actually, for better packing, let's sort by (x, y, z)
            
            # Find the best region: smallest (x, y, z) that fits
            best_idx = -1
            best_pos = (float('inf'), float('inf'), float('inf'))
            
            for i, region in enumerate(free_regions):
                rx, ry, rz, rw, rh, rd = region
                if rw >= w and rh >= h and rd >= d:
                    pos = (rx, ry, rz)
                    if pos < best_pos:
                        best_pos = pos
                        best_idx = i
            
            if best_idx >= 0:
                region = free_regions[best_idx]
                rx, ry, rz, rw, rh, rd = region
                
                # Remove the region
                free_regions.pop(best_idx)
                
                # Split and add new regions
                new_regions = split_region(region, (w, h, d))
                free_regions.extend(new_regions)
                
                return (bi, rx, ry, rz)
        
        return None
    
    # Process boxes
    placements = []
    
    for box in boxes:
        w, h, d = box
        
        # Try to place in existing bins
        result = find_and_place(bins, (w, h, d))
        
        if result is not None:
            bi, x, y, z = result
            placements.append((bi, x, y, z, w, h, d))
        else:
            # Open a new bin
            bi = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            
            # Place at origin
            x, y, z = 0, 0, 0
            
            # Split the initial region
            new_regions = split_region((0, 0, 0, W, H, D), (w, h, d))
            bins[bi] = new_regions
            
            placements.append((bi, x, y, z, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
