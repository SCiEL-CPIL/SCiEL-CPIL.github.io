import numpy as np
from math import ceil
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    
    # 1. Generate candidate slab heights
    # We consider all unique dimensions from all boxes as potential slab heights.
    # This ensures that at least one box can fit perfectly in height for many slabs.
    unique_dims = set()
    for w, h, d in boxes:
        unique_dims.add(w)
        unique_dims.add(h)
        unique_dims.add(d)
    
    # Add some intermediate values to potentially fill gaps better
    candidates = set(unique_dims)
    # Add H itself
    candidates.add(H)
    
    # Filter candidates that are valid slab heights (<= H)
    candidates = [c for c in candidates if 0 < c <= H]
    
    # To limit computation, if there are too many candidates, sample them
    # Sort by frequency or just take a subset. Let's take a diverse set.
    # For 60 boxes, there are at most 180 unique dims. This is manageable.
    # We will try all unique dims that are <= H.
    
    best_placements = None
    best_volume = -1
    
    # Pre-compute all rotations for each box to speed up inner loops
    # rot[i] is a list of (x, y, z) tuples for box i
    rotations = []
    for w, h, d in boxes:
        perms = set(permutations([w, h, d]))
        rotations.append(list(perms))
        
    # We will try each candidate slab height
    # To avoid excessive time, we might limit the number of slab heights tried if there are many
    # But 180 is small enough.
    
    for slab_h in candidates:
        # Determine number of slabs
        # We can have slabs of height slab_h, and potentially a last smaller slab
        # But the directive says "discretizes the container height into fixed-height horizontal slabs".
        # Usually this implies uniform slabs. If H is not divisible by slab_h, we can have
        # floor(H/slab_h) slabs of height slab_h, and one remainder slab.
        # However, to maximize volume, we should probably just use full slabs of height slab_h
        # and ignore the remainder if it's too small to fit anything, or handle it as a separate case.
        # Let's stick to full slabs of height slab_h for simplicity and consistency with "fixed-height".
        # Actually, if we have a remainder, we can pack it separately.
        
        n_full_slabs = int(H // slab_h)
        remainder_h = H - n_full_slabs * slab_h
        
        # If we have a remainder, we might treat it as another slab of height remainder_h
        # But the rotation logic depends on slab_h.
        # Let's just use n_full_slabs slabs of height slab_h.
        # This leaves the remainder empty. This is a conservative approach.
        # Alternatively, we can pack the remainder with a different heuristic.
        # Given the 120s limit, let's stick to uniform slabs first.
        
        if n_full_slabs == 0:
            continue
            
        # For each box, determine best rotation for this slab height
        # We want to minimize the height used (y) to leave room, but also consider footprint.
        # For now, pick the rotation with the smallest y that fits.
        # If multiple, pick the one with smallest area?
        
        box_rotations = []
        for i in range(len(boxes)):
            best_rot = None
            for (a, b, c) in rotations[i]:
                if b <= slab_h:
                    if best_rot is None or b < best_rot[1]:
                        best_rot = (a, b, c)
            if best_rot is not None:
                box_rotations.append((i, best_rot))
                
        if not box_rotations:
            continue
            
        # Sort boxes for 2D packing
        # Heuristic: Sort by height (y) descending, then by max(x,z) descending
        # This helps in shelf packing by placing "tall" (in 2D sense, i.e., wide in z or x) items first?
        # Actually, for shelf packing, we usually sort by height (z in 2D plane) descending.
        # Here, the 2D plane is (x, z). The "height" of the shelf is z.
        # So we should sort by c (z-extent) descending.
        
        box_rotations.sort(key=lambda x: x[1][2], reverse=True)
        
        # Initialize placements
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        
        total_volume = 0
        
        # Pack each slab
        for slab_idx in range(n_full_slabs):
            y_start = slab_idx * slab_h
            
            # 2D Shelf Packing in (x, z) plane
            # Shelves are horizontal strips of width W and variable height (z).
            # We maintain a list of shelves: (z_start, x_cursor, shelf_height)
            # shelf_height is the max z-extent of boxes in this shelf.
            
            shelves = []
            
            # We iterate through the sorted boxes.
            # To improve packing, we might want to try multiple sorting orders?
            # For now, let's stick to one good order: sort by z-extent (c) descending.
            
            for i, (a, b, c) in box_rotations:
                # Try to place box (a, b, c) in one of the existing shelves
                placed = False
                best_shelf_idx = -1
                best_x_cursor = -1
                best_z_start = -1
                best_shelf_h = -1
                
                # Find the best shelf (smallest gap or best fit)
                # Simple strategy: First Fit Decreasing by z-extent
                # We already sorted by c descending.
                # Let's try to find a shelf where c <= shelf_height and x_cursor + a <= W.
                # Among those, pick the one with smallest shelf_height? Or smallest x_cursor?
                # Let's try Best Fit: minimize (shelf_height - c) + (W - (x_cursor + a))?
                # For simplicity, let's use First Fit: first shelf that works.
                
                for s_idx, (z_start, x_cursor, s_h) in enumerate(shelves):
                    if c <= s_h and x_cursor + a <= W:
                        # Can place here
                        # Check if this is a "better" fit than previous?
                        # Let's just take the first one.
                        best_shelf_idx = s_idx
                        best_x_cursor = x_cursor
                        best_z_start = z_start
                        best_shelf_h = s_h
                        break
                        
                if best_shelf_idx != -1:
                    # Place in existing shelf
                    placements[i] = (0, best_x_cursor, y_start, best_z_start, a, b, c)
                    # Update shelf x_cursor
                    shelves[best_shelf_idx] = (best_z_start, best_x_cursor + a, best_shelf_h)
                    total_volume += a * b * c
                else:
                    # Open new shelf
                    # New shelf starts at z_start = sum of heights of previous shelves
                    if shelves:
                        z_start = shelves[-1][0] + shelves[-1][2]
                    else:
                        z_start = 0
                    
                    if z_start + c <= D:
                        placements[i] = (0, 0, y_start, z_start, a, b, c)
                        shelves.append((z_start, a, c))
                        total_volume += a * b * c
                    # If it doesn't fit in depth, box is left out in this slab
                    # (It might fit in another slab? No, all slabs have same dimensions WxD)
                    # So it's left out entirely.
        
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
