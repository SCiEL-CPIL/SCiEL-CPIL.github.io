import time
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # Preprocess boxes
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        vol = w * h * d
        box_data.append((w, h, d, i, vol))
        
    # Cache rotations
    # Since boxes are small (60), we can compute on the fly or cache.
    # Caching by (w,h,d) tuple is efficient.
    rot_cache = {}
    def get_rotations(w, h, d):
        key = (w, h, d)
        if key not in rot_cache:
            rots = set()
            for p in permutations([w, h, d]):
                rots.add(p)
            rot_cache[key] = list(rots)
        return rot_cache[key]
    
    best_volume = -1
    best_placements = None
    
    # Define sorting strategies
    # We will run the packing algorithm for each strategy
    # and keep the best result.
    
    strategies = [
        ("vol_desc", lambda b: -b[4]),
        ("max_dim_desc", lambda b: -max(b[0], b[1], b[2])),
        ("min_dim_desc", lambda b: -min(b[0], b[1], b[2])),
        ("vol_asc", lambda b: b[4]),
        ("max_dim_asc", lambda b: max(b[0], b[1], b[2])),
        ("min_dim_asc", lambda b: min(b[0], b[1], b[2])),
    ]
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_boxes = sorted(box_data, key=key_func)
        
        # Packing State
        placed = [None] * n
        total_vol = 0
        z_offset = 0
        remaining_indices = [b[3] for b in sorted_boxes]
        
        box_by_idx = {b[3]: b for b in box_data}
        
        while remaining_indices and z_offset < D:
            if time.time() > deadline:
                break
                
            # Skyline for current layer
            # List of (x_start, x_end, height)
            skyline = [(0, W, 0)]
            layer_max_h = 0
            boxes_to_remove = []
            
            # Process boxes in the sorted order
            for idx in remaining_indices:
                if time.time() > deadline:
                    break
                    
                w, h, d, _, vol = box_by_idx[idx]
                rots = get_rotations(w, h, d)
                
                best_pos = None
                best_cost = float('inf')
                best_rot = None
                
                for rw, rh, rd in rots:
                    if rd > D - z_offset:
                        continue
                    
                    # Candidate X positions
                    # 0 and start of each block
                    candidates = [0]
                    for bs, be, bh in skyline:
                        if bs > 0:
                            candidates.append(bs)
                    
                    # Remove duplicates and sort
                    candidates = sorted(set(candidates))
                    
                    for x in candidates:
                        if x + rw > W:
                            continue
                        
                        # Calculate max height in [x, x+rw)
                        max_h = 0
                        for bs, be, bh in skyline:
                            if x < be and x + rw > bs:
                                if bh > max_h:
                                    max_h = bh
                            
                        if max_h + rh > H:
                            continue
                            
                        # Cost: Primary is resulting height, Secondary is width (prefer narrow)
                        cost = (max_h + rh, rw)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, max_h)
                            best_rot = (rw, rh, rd)
                            
                if best_pos is not None:
                    x, y = best_pos
                    rw, rh, rd = best_rot
                    
                    placed[idx] = (x, y, z_offset, rw, rh, rd)
                    total_vol += vol
                    boxes_to_remove.append(idx)
                    
                    # Update Skyline
                    new_block_h = y + rh
                    updated_skyline = []
                    for bs, be, bh in skyline:
                        if be <= x or bs >= x + rw:
                            updated_skyline.append((bs, be, bh))
                        else:
                            if bs < x:
                                updated_skyline.append((bs, x, bh))
                            if be > x + rw:
                                updated_skyline.append((x + rw, be, bh))
                    
                    updated_skyline.append((x, x + rw, new_block_h))
                    updated_skyline.sort(key=lambda t: t[0])
                    
                    # Merge adjacent blocks with same height
                    merged = []
                    for block in updated_skyline:
                        if merged and merged[-1][1] == block[0] and merged[-1][2] == block[2]:
                            merged[-1] = (merged[-1][0], block[1], block[2])
                        else:
                            merged.append(block)
                            
                    skyline = merged
                    layer_max_h = max(layer_max_h, new_block_h)
                    
            for idx in boxes_to_remove:
                remaining_indices.remove(idx)
                
            z_offset += layer_max_h
            
        # Construct result for this strategy
        current_placements = []
        for i in range(n):
            if placed[i] is not None:
                x, y, z, rw, rh, rd = placed[i]
                current_placements.append((0, x, y, z, rw, rh, rd))
            else:
                w, h, d, _, _ = box_by_idx[i]
                current_placements.append((-1, 0, 0, 0, w, h, d))
                
        if total_vol > best_volume:
            best_volume = total_vol
            best_placements = current_placements
            
    if best_placements is None:
        best_placements = []
        for i in range(n):
            w, h, d = boxes[i]
            best_placements.append((-1, 0, 0, 0, w, h, d))
            
    return best_placements
