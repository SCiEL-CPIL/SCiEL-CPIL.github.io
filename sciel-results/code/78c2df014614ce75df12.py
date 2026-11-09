"""3D Bin Packing with BSP-tree based empty-space management."""
import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes: larger volume first, then larger dims to fill corners better
    order = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True)
    
    placements = [None] * n
    
    # Time budget: 110s safety margin
    t0 = time.time()
    TIME_LIMIT = 110.0
    
    # BSP node: (x0,y0,z0, x1,y1,z1) representing empty region [x0,x1) x [y0,y1) x [z0,z1)
    # We'll use a list-based approach for speed
    
    # Each bin has a list of empty regions
    # Start with one bin, one empty region = full container
    bins = [[(0, 0, 0, W, H, D)]]  # bins[b] = list of empty regions
    
    # For each box in sorted order, try to place in existing bins
    # Use best-fit: pick the bin+region that minimizes leftover waste
    for idx in order:
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        w, h, d = boxes[idx]
        
        best_bin = -1
        best_region_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for b in range(len(bins)):
            regions = bins[b]
            for ri, (x0, y0, z0, x1, y1, z1) in enumerate(regions):
                rw = x1 - x0
                rh = y1 - y0
                rd = z1 - z0
                if rw < w or rh < h or rd < d:
                    continue
                # Try to place at the "corner" of this region (x0, y0, z0)
                # Score: prefer tighter fit (less leftover)
                leftover = rw * rh * rd - w * h * d
                # Prefer smaller leftover, and prefer regions that are more "compact"
                # Use a simple score: leftover volume
                if leftover < best_score:
                    best_score = leftover
                    best_bin = b
                    best_region_idx = ri
                    best_pos = (x0, y0, z0)
        
        if best_bin == -1:
            # Need a new bin
            best_bin = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            best_region_idx = 0
            best_pos = (0, 0, 0)
        
        # Place the box at best_pos in best_bin
        x, y, z = best_pos
        region = bins[best_bin][best_region_idx]
        x0, y0, z0, x1, y1, z1 = region
        
        # Remove the old region and add split regions
        # The box occupies [x, x+w) x [y, y+h) x [z, z+d)
        # Since we place at corner (x0, y0, z0), the box is at (x0, y0, z0)
        # Wait, we placed at (x0, y0, z0) of the region. Let me re-check.
        # Actually best_pos = (x0, y0, z0) of the region.
        
        # Split the region around the box
        new_regions = []
        # Region: [x0, x1) x [y0, y1) x [z0, z1)
        # Box at (x0, y0, z0) with dims (w, h, d)
        # Remaining space:
        # 1. To the right of box: [x0+w, x1) x [y0, y1) x [z0, z1)  (if x0+w < x1)
        if x0 + w < x1:
            new_regions.append((x0 + w, y0, z0, x1, y1, z1))
        # 2. Above box (y direction): [x0, x0+w) x [y0+h, y1) x [z0, z1)  (if y0+h < y1)
        if y0 + h < y1:
            new_regions.append((x0, y0 + h, z0, x0 + w, y1, z1))
        # 3. Behind box (z direction): [x0, x0+w) x [y0, y0+h) x [z0+d, z1)  (if z0+d < z1)
        if z0 + d < z1:
            new_regions.append((x0, y0, z0 + d, x0 + w, y0 + h, z1))
        
        bins[best_bin] = new_regions
        placements[idx] = (best_bin, x0, y0, z0, w, h, d)
    
    # Handle any unplaced boxes (shouldn't happen if we always create new bins)
    for idx in range(n):
        if placements[idx] is None:
            b = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            w, h, d = boxes[idx]
            # Place at origin
            new_regions = []
            if w < W:
                new_regions.append((w, 0, 0, W, H, D))
            if h < H:
                new_regions.append((0, h, 0, w, H, D))
            if d < D:
                new_regions.append((0, 0, d, w, h, D))
            bins[b] = new_regions
            placements[idx] = (b, 0, 0, 0, w, h, d)
    
    # Renumber bins to be consecutive starting from 0
    # Map old bin labels to new consecutive labels
    bin_labels = sorted(set(p[0] for p in placements))
    label_map = {old: new for new, old in enumerate(bin_labels)}
    
    result = []
    for p in placements:
        b, x, y, z, w, h, d = p
        result.append((label_map[b], x, y, z, w, h, d))
    
    return result
