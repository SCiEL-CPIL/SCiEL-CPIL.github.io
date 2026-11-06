import time

def load(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    placed = []  # (x, y, z, w, h, d, orig_idx)
    
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((w, d, h))
        rots.add((d, h, w))
        rots.add((h, w, d))
        return list(rots)

    def is_valid_2d(x, y, w, h, obstacles):
        # obstacles: list of (ox, oy, ow, oh)
        if x + w > W or y + h > H:
            return False
        for ox, oy, ow, oh in obstacles:
            if x < ox + ow and ox < x + w and \
               y < oy + oh and oy < y + h:
                return False
        return True

    def find_best_position(w, h, d):
        best_pos = None
        best_score = float('inf')
        
        # Candidate z levels
        z_candidates = set([0])
        for p in placed:
            z_candidates.add(p[2])
            z_candidates.add(p[2] + p[5])
        
        for z in sorted(z_candidates):
            if z + d > D:
                continue
                
            # Determine obstacles for this z level
            # An obstacle is a box that overlaps in z
            obstacles = []
            for p in placed:
                pz, pd = p[2], p[5]
                if z < pz + pd and pz < z + d:
                    # Overlap in z
                    obstacles.append((p[0], p[1], p[3], p[4])) # x, y, w, h
            
            # Candidate x, y
            x_candidates = set([0])
            y_candidates = set([0])
            for ox, oy, ow, oh in obstacles:
                x_candidates.add(ox)
                x_candidates.add(ox + ow)
                y_candidates.add(oy)
                y_candidates.add(oy + oh)
                
            # Also add coordinates from ALL placed boxes?
            # No, only obstacles matter for 2D overlap.
            # But we might want to align with non-obstacles? No, that creates gaps.
            
            # Check all combinations of x, y candidates
            found_x_y = None
            best_xy_score = float('inf')
            
            for y in sorted(y_candidates):
                if y + h > H:
                    continue
                for x in sorted(x_candidates):
                    if is_valid_2d(x, y, w, h, obstacles):
                        score_xy = y * 1000 + x
                        if score_xy < best_xy_score:
                            best_xy_score = score_xy
                            found_x_y = (x, y)
            
            if found_x_y:
                x, y = found_x_y
                score = z * 1000000 + best_xy_score
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
        
        return best_pos

    start_time = time.time()
    timeout = 110
    
    for i, (orig_idx, box) in indexed_boxes:
        if time.time() - start_time > timeout:
            break
            
        w, h, d = box
        rotations = get_rotations(w, h, d)
        
        best_rot_pos = None
        best_rot_score = float('inf')
        
        for rw, rh, rd in rotations:
            pos = find_best_position(rw, rh, rd)
            if pos:
                x, y, z = pos
                score = z * 1000000 + y * 1000 + x
                if score < best_rot_score:
                    best_rot_score = score
                    best_rot_pos = (x, y, z, rw, rh, rd)
        
        if best_rot_pos:
            x, y, z, rw, rh, rd = best_rot_pos
            placed.append((x, y, z, rw, rh, rd, orig_idx))
    
    result = [None] * len(boxes)
    for x, y, z, w, h, d, orig_idx in placed:
        result[orig_idx] = (0, x, y, z, w, h, d)
    
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
            
    return result
