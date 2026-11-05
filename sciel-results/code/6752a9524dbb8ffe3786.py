import time
import itertools
import random

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


def check_overlap(px, py, pz, rw, rh, rd, placed_boxes):
    """Check if a box at position with given dimensions overlaps any placed box."""
    for pbox in placed_boxes:
        p_x, p_y, p_z, p_w, p_h, p_d = pbox
        if px < p_x + p_w and p_x < px + rw and \
           py < p_y + p_h and p_y < py + rh and \
           pz < p_z + p_d and p_z < pz + rd:
            return True
    return False

def try_pack(boxes, container, sort_key, deadline=None):
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
        if deadline and time.time() > deadline:
            # Mark remaining as left out
            for b, i in indexed_boxes:
                if results[i] is None:
                    results[i] = (-1, 0, 0, 0, b[0], b[1], b[2])
            break
            
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
            if pz + min(bd, bh, bw) > D:
                continue
            for py in sorted_ys:
                if found:
                    break
                if py + min(bd, bh, bw) > H:
                    continue
                for px in sorted_xs:
                    # Check all rotations at this position
                    for rot in rotations:
                        rw, rh, rd = rot
                        
                        # Check bounds
                        if px + rw > W or py + rh > H or pz + rd > D:
                            continue
                        
                        # Check overlap
                        if not check_overlap(px, py, pz, rw, rh, rd, placed_boxes):
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
            
    # Mark any unprocessed boxes as left out
    for b, i in indexed_boxes:
        if results[i] is None:
            results[i] = (-1, 0, 0, 0, b[0], b[1], b[2])
            
    return results, total_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
        lambda b: -(b[0] * b[1]), # Area of largest face descending
    ]
    
    best_results = None
    best_volume = -1
    
    for strategy in sort_strategies:
        if time.time() > deadline - 20: # Safety margin
            break
            
        results, vol = try_pack(boxes, container, strategy, deadline)
        if vol > best_volume:
            best_volume = vol
            best_results = results
            
    # Try a second pass: for boxes left out, try to place them in remaining space
    if best_results is not None and time.time() < deadline - 15:
        # Rebuild placement from best_results
        placed_boxes = []
        for r in best_results:
            if r[0] == 0:
                placed_boxes.append((r[1], r[2], r[3], r[4], r[5], r[6]))
        
        # Collect left-out boxes
        left_out = [(boxes[i], i) for i in range(n) if best_results[i][0] == -1]
        
        if left_out:
            # Sort left-out boxes by volume descending
            left_out.sort(key=lambda x: -(x[0][0] * x[0][1] * x[0][2]))
            
            xs = {0, W}
            ys = {0, H}
            zs = {0, D}
            for pbox in placed_boxes:
                xs.add(pbox[0])
                xs.add(pbox[0] + pbox[3])
                ys.add(pbox[1])
                ys.add(pbox[1] + pbox[4])
                zs.add(pbox[2])
                zs.add(pbox[2] + pbox[5])
            
            improved = False
            for box, idx in left_out:
                if time.time() > deadline - 5:
                    break
                    
                bw, bh, bd = box
                rotations = get_rotations(bw, bh, bd)
                
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
                            for rot in rotations:
                                rw, rh, rd = rot
                                if px + rw > W or py + rh > H or pz + rd > D:
                                    continue
                                if not check_overlap(px, py, pz, rw, rh, rd, placed_boxes):
                                    # Place it
                                    placed_boxes.append((px, py, pz, rw, rh, rd))
                                    best_results[idx] = (0, px, py, pz, rw, rh, rd)
                                    xs.add(px)
                                    xs.add(px + rw)
                                    ys.add(py)
                                    ys.add(py + rh)
                                    zs.add(pz)
                                    zs.add(pz + rd)
                                    best_volume += rw * rh * rd
                                    improved = True
                                    found = True
                                    break
                            
            if not improved:
                pass  # No improvement from second pass
    
    if best_results is None:
        # Fallback: return all left out
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
