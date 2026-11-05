import time
import itertools

def get_rotations(w, h, d):
    """
    Returns a list of unique (w, h, d) tuples for all 6 rotations.
    """
    rotations = [
        (w, h, d),
        (w, d, h),
        (h, w, d),
        (h, d, w),
        (d, w, h),
        (d, h, w)
    ]
    # Remove duplicates
    seen = set()
    unique_rotations = []
    for r in rotations:
        if r not in seen:
            seen.add(r)
            unique_rotations.append(r)
    return unique_rotations

def try_pack(boxes, container, sort_key):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    # Create list of (box, original_index)
    indexed_boxes = [(boxes[i], i) for i in range(n)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0]))
    
    placed_boxes = [] # List of (x, y, z, w, h, d)
    results = [None] * n
    total_volume = 0
    
    # Sets of unique coordinates
    xs = {0, W}
    ys = {0, H}
    zs = {0, D}
    
    for box, idx in indexed_boxes:
        bw, bh, bd = box
        best_pos = None
        best_rot = None
        
        rotations = get_rotations(bw, bh, bd)
        
        # Sort coordinates to iterate in order of preference (z, y, x)
        sorted_xs = sorted(xs)
        sorted_ys = sorted(ys)
        sorted_zs = sorted(zs)
        
        found = False
        for pz in sorted_zs:
            if found:
                break
            for py in sorted_ys:
                if found:
                    break
                for px in sorted_xs:
                    # Check all rotations at this position
                    for rot in rotations:
                        rw, rh, rd = rot
                        
                        # Check bounds
                        if px + rw > W or py + rh > H or pz + rd > D:
                            continue
                        
                        # Check overlap
                        overlap = False
                        for pbox in placed_boxes:
                            p_x, p_y, p_z, p_w, p_h, p_d = pbox
                            if px < p_x + p_w and p_x < px + rw and \
                               py < p_y + p_h and p_y < py + rh and \
                               pz < p_z + p_d and p_z < pz + rd:
                                overlap = True
                                break
                        
                        if not overlap:
                            # First valid position in sorted order is the best
                            best_pos = (px, py, pz)
                            best_rot = rot
                            found = True
                            break
                            
        if best_pos is not None:
            bx, by, bz = best_pos
            bw_r, bh_r, bd_r = best_rot
            # Place the box
            placed_boxes.append((bx, by, bz, bw_r, bh_r, bd_r))
            total_volume += bw_r * bh_r * bd_r
            results[idx] = (0, bx, by, bz, bw_r, bh_r, bd_r)
            
            # Update coordinate sets
            xs.add(bx)
            xs.add(bx + bw_r)
            ys.add(by)
            ys.add(by + bh_r)
            zs.add(bz)
            zs.add(bz + bd_r)
        else:
            results[idx] = (-1, 0, 0, 0, bw, bh, bd)
            
    return results, total_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    n = len(boxes)
    if n == 0:
        return []
        
    # Define different sorting strategies
    sort_strategies = [
        lambda b: -(b[0] * b[1] * b[2]), # Volume descending
        lambda b: -max(b), # Max dimension descending
        lambda b: -(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]), # Surface area descending
        lambda b: -min(b), # Min dimension descending
    ]
    
    best_results = None
    best_volume = -1
    
    for strategy in sort_strategies:
        if time.time() > deadline - 10: # Safety margin
            break
            
        results, vol = try_pack(boxes, container, strategy)
        if vol > best_volume:
            best_volume = vol
            best_results = results
            
    if best_results is None:
        # Fallback: return all left out
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
