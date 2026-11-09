import time

def load(boxes, container):
    start_time = time.time()
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
        
    # Sort indices by volume descending
    indices = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    placements = [None] * n
    x, y, z = 0, 0, 0
    row_h, layer_d = 0, 0
    
    def get_rotations(w, h, d):
        return [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]

    for idx in indices:
        if time.time() - start_time > 110:
            # Fill remaining with -1
            for i in range(n):
                if placements[i] is None:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            return placements

        w, h, d = boxes[idx]
        rots = get_rotations(w, h, d)
        
        placed = False
        
        # 1. Try to place at current (x, y, z)
        best_rot = None
        best_score = None
        
        for rw, rh, rd in rots:
            if x + rw <= W and y + rh <= H and z + rd <= D:
                score = (max(row_h, rh), rd, rw)
                if best_score is None or score < best_score:
                    best_score = score
                    best_rot = (rw, rh, rd)
        
        if best_rot:
            rw, rh, rd = best_rot
            placements[idx] = (0, x, y, z, rw, rh, rd)
            x += rw
            row_h = max(row_h, rh)
            layer_d = max(layer_d, rd)
            placed = True
        
        # 2. If not placed, try next shelf
        if not placed:
            new_y = y + row_h
            # Quick check
            if new_y < H:
                x_temp = 0
                row_h_temp = 0
                best_rot = None
                best_score = None
                for rw, rh, rd in rots:
                    if x_temp + rw <= W and new_y + rh <= H and z + rd <= D:
                        score = (rh, rd, rw)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_rot = (rw, rh, rd)
                
                if best_rot:
                    rw, rh, rd = best_rot
                    placements[idx] = (0, x_temp, new_y, z, rw, rh, rd)
                    x = x_temp + rw
                    y = new_y
                    row_h = rh
                    layer_d = max(layer_d, rd)
                    placed = True
        
        # 3. If not placed, try next layer
        if not placed:
            new_z = z + layer_d
            if new_z < D:
                x_temp = 0
                y_temp = 0
                row_h_temp = 0
                layer_d_temp = 0
                best_rot = None
                best_score = None
                for rw, rh, rd in rots:
                    if x_temp + rw <= W and y_temp + rh <= H and new_z + rd <= D:
                        score = (rh, rd, rw)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_rot = (rw, rh, rd)
                
                if best_rot:
                    rw, rh, rd = best_rot
                    placements[idx] = (0, x_temp, y_temp, new_z, rw, rh, rd)
                    x = x_temp + rw
                    y = y_temp
                    z = new_z
                    row_h = rh
                    layer_d = rd
                    placed = True
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    return placements
