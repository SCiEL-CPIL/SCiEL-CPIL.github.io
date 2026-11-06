import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    indexed_boxes = list(enumerate(boxes))
    
    strategies = [
        lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])),
        lambda x: (-max(x[1]), -(x[1][0]*x[1][1]*x[1][2])),
        lambda x: (-(2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][2]*x[1][0])), -max(x[1])),
        lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -(min(x[1])), -max(x[1]))
    ]
    
    best_placements = None
    best_volume = -1
    
    for strategy in strategies:
        sorted_boxes = sorted(indexed_boxes, key=strategy)
        
        # Store occupied boxes in a numpy array for fast vectorized overlap checks
        # Columns: x, y, z, w, h, d, idx
        occupied_np = np.empty((0, 7), dtype=int)
        placements = [None] * len(boxes)
        current_volume = 0
        
        def get_candidates(w, h, d):
            xs = {0}
            ys = {0}
            zs = {0}
            if len(occupied_np) > 0:
                ox = occupied_np[:, 0]
                oy = occupied_np[:, 1]
                oz = occupied_np[:, 2]
                ow = occupied_np[:, 3]
                oh = occupied_np[:, 4]
                od = occupied_np[:, 5]
                
                # x candidates
                xs.update((ox + ow)[ox + ow <= W - w].tolist())
                xs.update((ox - w)[ox - w >= 0].tolist())
                
                # y candidates
                ys.update((oy + oh)[oy + oh <= H - h].tolist())
                ys.update((oy - h)[oy - h >= 0].tolist())
                
                # z candidates
                zs.update((oz + od)[oz + od <= D - d].tolist())
                zs.update((oz - d)[oz - d >= 0].tolist())
            
            xs = [x for x in xs if 0 <= x <= W - w]
            ys = [y for y in ys if 0 <= y <= H - h]
            zs = [z for z in zs if 0 <= z <= D - d]
            
            xs.sort()
            ys.sort()
            zs.sort()
            
            if not xs: xs = [0]
            if not ys: ys = [0]
            if not zs: zs = [0]
            
            return xs, ys, zs
        
        def is_valid(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            if len(occupied_np) == 0:
                return True
            
            # Vectorized overlap check
            # Overlap if NOT (x+w <= ox OR ox+ow <= x)
            # AND NOT (y+h <= oy OR oy+oh <= y)
            # AND NOT (z+d <= oz OR oz+od <= z)
            
            ox = occupied_np[:, 0]
            oy = occupied_np[:, 1]
            oz = occupied_np[:, 2]
            ow = occupied_np[:, 3]
            oh = occupied_np[:, 4]
            od = occupied_np[:, 5]
            
            sep_x = (x + w <= ox) | (ox + ow <= x)
            sep_y = (y + h <= oy) | (oy + oh <= y)
            sep_z = (z + d <= oz) | (oz + od <= z)
            
            # If any dimension is separated, there is no overlap in that dimension.
            # We need overlap in ALL dimensions to be invalid.
            # So if ANY dimension is separated, it's valid (no overlap).
            # If NO dimension is separated, it's invalid (overlap).
            
            # overlap_exists = ~sep_x & ~sep_y & ~sep_z
            # valid = not np.any(overlap_exists)
            
            # More efficient:
            # If np.any(sep_x) is True, it doesn't mean valid, it means SOME boxes are separated.
            # We need to check if there is ANY box that is NOT separated in x AND NOT separated in y AND NOT separated in z.
            
            overlap_mask = (~sep_x) & (~sep_y) & (~sep_z)
            return not np.any(overlap_mask)
        
        def place_box(box_idx, w, h, d):
            nonlocal occupied_np, current_volume
            for perm in permutations([w, h, d]):
                pw, ph, pd = perm
                xs, ys, zs = get_candidates(pw, ph, pd)
                for x in xs:
                    for y in ys:
                        for z in zs:
                            if is_valid(x, y, z, pw, ph, pd):
                                # Place the box
                                new_box = np.array([[x, y, z, pw, ph, pd, box_idx]])
                                occupied_np = np.vstack([occupied_np, new_box])
                                placements[box_idx] = (0, x, y, z, pw, ph, pd)
                                current_volume += pw * ph * pd
                                return True
            return False
        
        for box_idx, (w, h, d) in sorted_boxes:
            place_box(box_idx, w, h, d)
            if placements[box_idx] is None:
                placements[box_idx] = (-1, 0, 0, 0, w, h, d)
        
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = placements[:]
            
    return best_placements
