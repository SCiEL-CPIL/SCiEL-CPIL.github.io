import numpy as np
from math import ceil

def load(boxes, container):
    W, H, D = container
    
    # Get all unique heights from all rotations
    all_dims = []
    for w, h, d in boxes:
        all_dims.append((w, h, d))
        all_dims.append((h, w, d))
        all_dims.append((w, d, h))
    
    # Candidate slab heights: unique heights, plus some quantiles
    heights = sorted(set(dim[1] for dim in all_dims))
    if not heights:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try a few slab height candidates
    candidates = set()
    candidates.add(heights[0])
    candidates.add(heights[len(heights)//2])
    candidates.add(heights[-1])
    candidates.add(H)
    # Also try a few intermediate
    for q in [0.25, 0.75]:
        idx = int(len(heights) * q)
        if idx < len(heights):
            candidates.add(heights[idx])
    
    best_placements = None
    best_volume = -1
    
    for slab_h in sorted(candidates):
        if slab_h <= 0 or slab_h > H:
            continue
        n_slabs = ceil(H / slab_h)
        if n_slabs == 0:
            continue
        
        # For each box, find best rotation that fits in slab
        box_info = []
        for i, (w, h, d) in enumerate(boxes):
            # Try all 3 rotations: (w,h,d), (h,w,d), (w,d,h)
            # Height is the second element in my convention? Let me be careful.
            # Container: W (x), H (y), D (z). 
            # Wait, I need to be consistent. Let me define:
            # x-axis: width W
            # y-axis: height H  
            # z-axis: depth D
            # A box has dimensions (w, h, d) meaning it spans w in x, h in y, d in z.
            # Rotations: we can permute (w, h, d) to any permutation.
            # For slab packing, the "height" that matters is the y-dimension (h), since we slice by y.
            # So for each permutation (a, b, c) of (w, h, d), the box has:
            #   x-extent = a, y-extent = b, z-extent = c
            # It fits in a slab if b <= slab_h.
            # Within the slab, its 2D footprint is (a, c) in the (x, z) plane.
            
            best_rot = None
            for perm in [(w, h, d), (h, w, d), (w, d, h), (d, w, h), (h, d, w), (d, h, w)]:
                a, b, c = perm  # x-extent, y-extent, z-extent
                if b <= slab_h:
                    # This rotation fits
                    if best_rot is None or b < best_rot[1]:  # prefer smaller height
                        best_rot = (a, b, c)
            if best_rot is None:
                box_info.append((i, None))
            else:
                box_info.append((i, best_rot))
        
        # Separate boxes that fit and don't
        fitting = [(i, rot) for i, rot in box_info if rot is not None]
        not_fitting = [i for i, rot in box_info if rot is None]
        
        # Sort fitting boxes by y-extent descending (taller boxes first for shelf packing)
        fitting.sort(key=lambda x: x[1][1], reverse=True)
        
        # Initialize placements
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        
        # Pack each slab
        total_volume = 0
        for slab_idx in range(n_slabs):
            y_start = slab_idx * slab_h
            y_end = min(y_start + slab_h, H)
            actual_slab_h = y_end - y_start
            
            # 2D shelf packing in (x, z) plane with dimensions W x D
            shelves = []  # list of (z_start, x_cursor, max_z)
            
            for i, (a, b, c) in fitting:
                if b > actual_slab_h:
                    continue
                # Try to place on existing shelves
                placed = False
                for s_idx, (z_start, x_cursor, max_z) in enumerate(shelves):
                    if x_cursor + a <= W and c <= max_z:
                        # Place here
                        placements[i] = (0, x_cursor, y_start, z_start, a, b, c)
                        shelves[s_idx] = (z_start, x_cursor + a, max_z)
                        total_volume += a * b * c
                        placed = True
                        break
                if not placed:
                    # Open new shelf
                    if shelves:
                        z_start = shelves[-1][0] + shelves[-1][2]
                    else:
                        z_start = 0
                    if z_start + c <= D:
                        placements[i] = (0, 0, y_start, z_start, a, b, c)
                        shelves.append((z_start, a, c))
                        total_volume += a * b * c
        
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
