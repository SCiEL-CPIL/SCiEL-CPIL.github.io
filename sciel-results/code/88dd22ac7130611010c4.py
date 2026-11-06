import numpy as np
from math import ceil
import itertools

def load(boxes, container):
    W, H, D = container
    
    if not boxes:
        return []
        
    # Generate candidate slab heights
    # We want heights that correspond to box dimensions to avoid wasted space
    all_dims = []
    for w, h, d in boxes:
        all_dims.extend([w, h, d])
    
    # Unique heights from box dimensions
    candidate_heights = set(all_dims)
    # Add container height
    candidate_heights.add(H)
    # Add some intermediate values? Maybe just the unique box heights is enough.
    # But if H is not a multiple of any box height, we might have a partial slab at the top.
    # The loop handles partial slabs.
    
    # To avoid too many candidates, we can limit or just use all unique dims.
    # 60 boxes * 3 dims = 180 unique values max. This is fine.
    
    best_placements = None
    best_volume = -1
    
    # Pre-compute all 6 permutations for each box
    box_perms = []
    for w, h, d in boxes:
        perms = set()
        for p in itertools.permutations((w, h, d)):
            perms.add(p)
        box_perms.append(list(perms))
        
    for slab_h in sorted(candidate_heights):
        if slab_h <= 0 or slab_h > H:
            continue
            
        n_slabs = ceil(H / slab_h)
        if n_slabs == 0:
            continue
        
        # placements[i] = (bin, x, y, z, w, h, d)
        # bin 0 for loaded, -1 for left out
        current_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(len(boxes))]
        total_volume = 0
        
        # We need to track which boxes are already placed
        # Since slabs are independent in this heuristic (no box spans slabs),
        # we can process each slab independently and assign boxes.
        # However, a box can only be placed once.
        # So we need to iterate through slabs and pick boxes that haven't been placed yet.
        
        placed_boxes = set()
        
        for slab_idx in range(n_slabs):
            y_start = slab_idx * slab_h
            y_end = min(y_start + slab_h, H)
            actual_slab_h = y_end - y_start
            
            if actual_slab_h <= 0:
                continue
                
            # Filter boxes that fit in this slab height and haven't been placed
            candidates_for_slab = []
            for i in range(len(boxes)):
                if i in placed_boxes:
                    continue
                w, h, d = boxes[i]
                # Check if any rotation fits in actual_slab_h
                for perm in box_perms[i]:
                    a, b, c = perm # x, y, z
                    if b <= actual_slab_h:
                        candidates_for_slab.append((i, a, b, c))
                        break # Just one fit is enough to qualify, we'll pick best later?
                        # Actually, for packing, we might want to consider all rotations that fit.
                        # But usually, the one with smallest Y (height) is best for vertical space?
                        # No, Y is fixed to slab. In 2D, we care about X and Z.
                        # We should keep all valid rotations for the 2D packer to choose from?
                        # Or pick the "best" 2D footprint?
                        # Let's keep all valid rotations and let the 2D packer try the best fit.
                        # To keep it simple, let's just pick the rotation with smallest Z (shelf height) 
                        # or smallest Area? 
                        # Let's collect all valid rotations for this box in this slab.
            
            # Refine: Collect all valid (box_idx, x_dim, y_dim, z_dim) for this slab
            valid_rotations = []
            for i in range(len(boxes)):
                if i in placed_boxes:
                    continue
                for perm in box_perms[i]:
                    a, b, c = perm
                    if b <= actual_slab_h:
                        valid_rotations.append((i, a, b, c))
            
            if not valid_rotations:
                continue
                
            # We need to select a subset of these rotations to pack.
            # This is a complex selection problem.
            # Heuristic: Sort by Z (shelf height) descending, then X descending.
            # Then run Best Fit Shelf Packing.
            # If a box has multiple rotations, we might pick the one that fits best.
            # To simplify, let's assume we pick one rotation per box before packing.
            # Which one? The one with smallest Z, then smallest X?
            # Or the one with smallest Area?
            # Let's try: For each box, pick the rotation with smallest Z. If tie, smallest X.
            # This minimizes the "shelf height" required.
            
            box_best_rot = {}
            for i, a, b, c in valid_rotations:
                if i not in box_best_rot:
                    box_best_rot[i] = (a, b, c)
                else:
                    # Compare (a, b, c) with existing
                    # Prefer smaller c (z), then smaller a (x)
                    ea, eb, ec = box_best_rot[i]
                    if c < ec or (c == ec and a < ea):
                        box_best_rot[i] = (a, b, c)
            
            # Now we have a list of (idx, a, b, c)
            pack_list = [(i, a, b, c) for i, (a, b, c) in box_best_rot.items()]
            
            # Strategy 1: Shelves along X (width), stack along Z (depth)
            # Sort by Z (c) descending, then X (a) descending
            pack_list_s1 = sorted(pack_list, key=lambda x: (x[3], x[1]), reverse=True)
            
            # Strategy 2: Shelves along Z (depth), stack along X (width)
            # Sort by X (a) descending, then Z (c) descending
            pack_list_s2 = sorted(pack_list, key=lambda x: (x[1], x[3]), reverse=True)
            
            def pack_shelves_2d(list_items, W_limit, D_limit, orientation):
                # orientation: 0 = shelves along X (limit W), stack Z (limit D)
                #              1 = shelves along Z (limit D), stack X (limit W)
                # list_items: (idx, a, b, c) where a=x, c=z
                # For orientation 0: shelf_len = a, shelf_h = c
                # For orientation 1: shelf_len = c, shelf_h = a
                
                shelves = [] # (start_pos, shelf_height, cursor)
                current_max_pos = 0
                placed_in_slab = []
                
                for item in list_items:
                    idx, a, b, c = item
                    if orientation == 0:
                        len_dim = a
                        h_dim = c
                    else:
                        len_dim = c
                        h_dim = a
                        
                    # Try to fit in existing shelves
                    best_shelf_idx = -1
                    best_remaining = float('inf')
                    
                    for s_idx, (start_pos, shelf_h, cursor) in enumerate(shelves):
                        if h_dim <= shelf_h and cursor + len_dim <= (W_limit if orientation == 0 else D_limit):
                            remaining = (W_limit if orientation == 0 else D_limit) - (cursor + len_dim)
                            if remaining < best_remaining:
                                best_remaining = remaining
                                best_shelf_idx = s_idx
                                
                    if best_shelf_idx != -1:
                        # Place in best shelf
                        start_pos, shelf_h, cursor = shelves[best_shelf_idx]
                        new_cursor = cursor + len_dim
                        shelves[best_shelf_idx] = (start_pos, shelf_h, new_cursor)
                        
                        if orientation == 0:
                            # x = cursor, z = start_pos
                            x, z = cursor, start_pos
                        else:
                            # x = start_pos, z = cursor
                            x, z = start_pos, cursor
                            
                        placed_in_slab.append((idx, x, z, a, b, c))
                    else:
                        # Open new shelf
                        if orientation == 0:
                            if current_max_pos + h_dim <= D_limit:
                                shelves.append((current_max_pos, h_dim, len_dim))
                                current_max_pos += h_dim
                                # x = 0, z = start_pos (which is the old current_max_pos before update? No, start_pos is the start of this shelf)
                                # The start_pos for the new shelf is the current_max_pos BEFORE we added it?
                                # Yes.
                                # Wait, I updated current_max_pos. I need the start position.
                                # Let's re-structure:
                                pass # Logic fix needed below
                        else:
                            if current_max_pos + h_dim <= W_limit:
                                shelves.append((current_max_pos, h_dim, len_dim))
                                current_max_pos += h_dim
                                pass # Logic fix needed below
                            
                return placed_in_slab

            # The pack_shelves_2d function above has a bug in coordinate tracking.
            # Let's rewrite it cleanly.
            
            def run_packing(pack_list, orientation):
                # orientation 0: Shelf along X (limit W), Stack along Z (limit D)
                # orientation 1: Shelf along Z (limit D), Stack along X (limit W)
                
                shelves = [] # (start_pos, shelf_height, cursor)
                max_pos = 0
                placed = []
                
                for item in pack_list:
                    idx, a, b, c = item
                    if orientation == 0:
                        len_dim = a
                        h_dim = c
                        limit_len = W
                        limit_stack = D
                    else:
                        len_dim = c
                        h_dim = a
                        limit_len = D
                        limit_stack = W
                        
                    best_s = -1
                    best_rem = float('inf')
                    
                    for s_idx, (start, sh, cur) in enumerate(shelves):
                        if h_dim <= sh and cur + len_dim <= limit_len:
                            rem = limit_len - (cur + len_dim)
                            if rem < best_rem:
                                best_rem = rem
                                best_s = s_idx
                                
                    if best_s != -1:
                        start, sh, cur = shelves[best_s]
                        new_cur = cur + len_dim
                        shelves[best_s] = (start, sh, new_cur)
                        
                        if orientation == 0:
                            x, z = cur, start
                        else:
                            x, z = start, cur
                            
                        placed.append((idx, x, z, a, b, c))
                    else:
                        if max_pos + h_dim <= limit_stack:
                            # New shelf
                            start = max_pos
                            max_pos += h_dim
                            shelves.append((start, h_dim, len_dim))
                            
                            if orientation == 0:
                                x, z = 0, start
                            else:
                                x, z = start, 0
                                
                            placed.append((idx, x, z, a, b, c))
                return placed
            
            res1 = run_packing(pack_list_s1, 0)
            res2 = run_packing(pack_list_s2, 1)
            
            # Choose better result for this slab
            vol1 = sum(a*b*c for _, _, _, a, b, c in res1)
            vol2 = sum(a*b*c for _, _, _, a, b, c in res2)
            
            if vol1 >= vol2:
                best_res = res1
            else:
                best_res = res2
                
            # Update global placements
            for idx, x, z, a, b, c in best_res:
                current_placements[idx] = (0, x, y_start, z, a, b, c)
                placed_boxes.add(idx)
                total_volume += a * b * c
                
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = [p[:] for p in current_placements]
            
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
