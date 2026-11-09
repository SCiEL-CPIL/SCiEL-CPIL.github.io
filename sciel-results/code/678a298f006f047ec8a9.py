import numpy as np

def load(boxes, container):
    W, H, D = container
    
    # Strategy: Fixed-Height Shelf Packing
    # Sort boxes by z-dimension (height) descending to create tall shelves first.
    # This minimizes the number of shelves.
    
    strategies = [
        lambda b: b[2], # Sort by z-dim (height)
        lambda b: b[0]*b[1]*b[2], # Sort by volume
        lambda b: max(b), # Sort by max dim
    ]
    
    best_result = None
    best_volume = -1
    
    def solve_shelf_fixed(sort_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]), reverse=True)
        
        # Shelves: list of dicts
        # Each shelf: { 'z_start': float, 'z_end': float, 'height': float, 'boxes': [(x, y, w, h), ...] }
        shelves = []
        results = [None] * len(boxes)
        total_vol = 0
        
        for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
            placed = False
            
            # Try existing shelves
            for shelf in shelves:
                # Check if box height fits in this shelf's height
                if d_orig > shelf['height']:
                    continue
                
                # Try to place (w_orig, h_orig) in (W, H) plane of this shelf
                xs = [0]
                ys = [0]
                for (px, py, pw, ph) in shelf['boxes']:
                    xs.append(px + pw)
                    ys.append(py + ph)
                
                xs = sorted(set(x for x in xs if x < W))
                ys = sorted(set(y for y in ys if y < H))
                
                orientations_xy = [
                    (w_orig, h_orig),
                    (h_orig, w_orig)
                ]
                
                found_pos = None
                for ow, oh in orientations_xy:
                    if ow > W or oh > H:
                        continue
                    for cy in ys:
                        if cy + oh > H:
                            break
                        for cx in xs:
                            if cx + ow > W:
                                break
                            
                            valid = True
                            for (px, py, pw, ph) in shelf['boxes']:
                                if not (cx + ow <= px or px + pw <= cx or
                                        cy + oh <= py or py + ph <= cy):
                                    valid = False
                                    break
                            
                            if valid:
                                found_pos = (cx, cy, ow, oh)
                                break
                        if found_pos:
                            break
                    if found_pos:
                        break
                
                if found_pos:
                    cx, cy, ow, oh = found_pos
                    shelf['boxes'].append((cx, cy, ow, oh))
                    results[orig_idx] = (0, cx, cy, shelf['z_start'], ow, oh, d_orig)
                    total_vol += ow * oh * d_orig
                    placed = True
                    break
            
            if not placed:
                # Create new shelf
                if d_orig > D:
                    results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
                    continue
                
                current_max_z = 0
                for s in shelves:
                    if s['z_end'] > current_max_z:
                        current_max_z = s['z_end']
                
                if current_max_z + d_orig > D:
                    results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
                    continue
                
                new_shelf = {
                    'z_start': current_max_z,
                    'z_end': current_max_z + d_orig,
                    'height': d_orig,
                    'boxes': []
                }
                shelves.append(new_shelf)
                
                # Place at (0,0)
                results[orig_idx] = (0, 0, 0, current_max_z, w_orig, h_orig, d_orig)
                new_shelf['boxes'].append((0, 0, w_orig, h_orig))
                total_vol += w_orig * h_orig * d_orig
        
        return results, total_vol
    
    for sort_key in strategies:
        res, vol = solve_shelf_fixed(sort_key)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result
