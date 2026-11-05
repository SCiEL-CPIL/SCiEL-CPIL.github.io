import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 unique rotations for a box
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)
    
    # Precompute valid rotations for each box
    box_rotations = []
    for i, (w, h, d) in enumerate(boxes):
        rots = get_rotations(w, h, d)
        # Filter rotations that fit in container
        valid_rots = [(rw, rh, rd) for rw, rh, rd in rots if rw <= W and rh <= H and rd <= D]
        if not valid_rots:
            # If no rotation fits, we can't place it, but keep original for output format
            valid_rots = [(w, h, d)] 
        box_rotations.append(valid_rots)
    
    # Define different sorting strategies
    # We want to try packing large boxes first, but different definitions of "large"
    # can lead to better packing for different box distributions.
    
    def sort_by_volume_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_min_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
        
    def sort_by_volume_max_dim_ratio(boxes):
        # Try to pack "blocky" boxes first?
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]), reverse=True)

    strategies = [
        sort_by_volume_desc,
        sort_by_max_dim_desc,
        sort_by_min_dim_desc,
        sort_by_volume_max_dim_ratio
    ]
    
    best_total_volume = 0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sort_func(boxes)
        placements = [None] * n
        placed_boxes = [] # List of (x, y, z, w, h, d, box_index)
        current_volume = 0
        
        def can_place(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            for (px, py, pz, pw, ph, pd, _) in placed_boxes:
                # Check for overlap in 3D
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True
            
        def find_best_position(rw, rh, rd):
            """
            Find the best position for a box with dimensions (rw, rh, rd).
            Candidates are generated from the 'extremal' points of placed boxes.
            """
            best_pos = None
            best_score = None
            
            # Generate candidate coordinates
            # X candidates: 0 and right faces of placed boxes
            x_candidates = set([0])
            y_candidates = set([0])
            z_candidates = set([0])
            
            for (px, py, pz, pw, ph, pd, _) in placed_boxes:
                if px + pw <= W:
                    x_candidates.add(px + pw)
                if py + ph <= H:
                    y_candidates.add(py + ph)
                if pz + pd <= D:
                    z_candidates.add(pz + pd)
            
            # Sort candidates to prioritize lower coordinates (compactness)
            x_candidates = sorted(x_candidates)
            y_candidates = sorted(y_candidates)
            z_candidates = sorted(z_candidates)
            
            for x in x_candidates:
                if x + rw > W:
                    continue
                for y in y_candidates:
                    if y + rh > H:
                        continue
                    for z in z_candidates:
                        if z + rd > D:
                            continue
                        if can_place(x, y, z, rw, rh, rd):
                            # Scoring: 
                            # 1. Minimize z (stack on bottom)
                            # 2. Minimize x+y (keep footprint compact)
                            # 3. Minimize x, then y
                            score = (z, x + y, x, y)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_pos = (x, y, z)
            
            return best_pos

        for box_idx in sorted_indices:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[box_idx]
            best_placement = None
            best_orientation = None
            best_pos_score = None
            
            for rw, rh, rd in box_rotations[box_idx]:
                pos = find_best_position(rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    pos_score = (z, x + y, x, y)
                    if best_pos_score is None or pos_score < best_pos_score:
                        best_pos_score = pos_score
                        best_placement = pos
                        best_orientation = (rw, rh, rd)
            
            if best_placement is not None:
                x, y, z = best_placement
                rw, rh, rd = best_orientation
                placements[box_idx] = (0, x, y, z, rw, rh, rd)
                placed_boxes.append((x, y, z, rw, rh, rd, box_idx))
                current_volume += rw * rh * rd
            else:
                placements[box_idx] = (-1, 0, 0, 0, w, h, d)
                
        # If we finished this strategy, check if it's better
        if current_volume > best_total_volume:
            best_total_volume = current_volume
            # Ensure all placements are filled (in case we broke early in a previous run, 
            # but here we finished the loop for this strategy)
            for i in range(n):
                if placements[i] is None:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            best_placements = placements[:]
            
    return best_placements
