"""Combined 3D Container Loading: BFDH with multi-orientation, priority ordering, and deadline guard."""
import time
import itertools
import math

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin vs 120s limit
    
    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]
        
        # Sort orientations: prefer those with smaller max dimension, then volume
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort by volume descending (Best-Fit Decreasing)
    box_data.sort(key=lambda x: x[2], reverse=True)
    
    def execute_placement(box_order):
        """Execute a placement strategy with a specific box ordering."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * n
        
        def get_candidate_axes():
            """Generate candidate coordinates from placed boxes and origin."""
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            return sorted(cx_set), sorted(cy_set), sorted(cz_set)
        
        def check_no_overlap(cx, cy, cz, bw, bh, bd):
            """Check if a box at (cx,cy,cz) with dims (bw,bh,bd) doesn't overlap any placed box."""
            for (px, py, pz, pw, ph, pd) in placed:
                if not ((cx + bw <= px) or (px + pw <= cx) or
                        (cy + bh <= py) or (py + ph <= cy) or
                        (cz + bd <= pz) or (pz + pd <= cz)):
                    return False
            return True
        
        def try_place_box(idx, orients, volume, ow, oh, od):
            """Try to place a box using best-fit with early termination."""
            cx_list, cy_list, cz_list = get_candidate_axes()
            
            # Precompute min dimensions for early pruning
            min_w = min(o[0] for o in orients)
            min_h = min(o[1] for o in orients)
            min_d = min(o[2] for o in orients)
            
            best_pos = None
            best_key = None
            best_fit_score = None
            
            # Iterate through candidate positions in priority order: minimize z, then y, then x
            for cz in cz_list:
                if cz + min_d > D:
                    continue
                for cy in cy_list:
                    if cy + min_h > H:
                        continue
                    for cx in cx_list:
                        if cx + min_w > W:
                            continue
                        
                        # Try each orientation
                        for (bw, bh, bd) in orients:
                            if cx + bw > W or cy + bh > H or cz + bd > D:
                                continue
                            
                            if not check_no_overlap(cx, cy, cz, bw, bh, bd):
                                continue
                            
                            # Found a valid position
                            # Best-fit: prefer positions where the box fits tightly
                            # Score: minimize the "wasted space" around the box
                            # Use contact score: how close the box is to boundaries and other boxes
                            contact_score = 0
                            # Check if touching bottom
                            if cz == 0:
                                contact_score += 1000
                            # Check if touching back
                            if cy == 0:
                                contact_score += 500
                            # Check if touching left
                            if cx == 0:
                                contact_score += 250
                            # Check if touching other boxes (approximate by checking proximity)
                            for (px, py, pz, pw, ph, pd) in placed:
                                # Check if adjacent in x
                                if abs((cx + bw) - px) < 1e-9 or abs((px + pw) - cx) < 1e-9:
                                    if not (cy + bh <= py or py + ph <= cy) and not (cz + bd <= pz or pz + pd <= cz):
                                        contact_score += 50
                                # Check if adjacent in y
                                if abs((cy + bh) - py) < 1e-9 or abs((py + ph) - cy) < 1e-9:
                                    if not (cx + bw <= px or px + pw <= cx) and not (cz + bd <= pz or pz + pd <= cz):
                                        contact_score += 30
                                # Check if adjacent in z
                                if abs((cz + bd) - pz) < 1e-9 or abs((pz + pd) - cz) < 1e-9:
                                    if not (cx + bw <= px or px + pw <= cx) and not (cz + bd <= py or py + ph <= cy):
                                        contact_score += 10
                            
                            # Composite key: prefer lower z, then higher contact score, then lower y, then lower x
                            key = (cz, -contact_score, cy, cx)
                            
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (cx, cy, cz, bw, bh, bd)
                            break  # No need to try other orientations for this position
                        
                        # Early termination: if we have a good position (touching bottom), stop
                        if best_pos is not None and best_key[0] == 0 and best_key[1] >= 1000:
                            return best_pos
            
            return best_pos
        
        # Process boxes in the given order
        for idx, orients, volume, ow, oh, od in box_order:
            if time.time() > deadline:
                break
            
            best_pos = try_place_box(idx, orients, volume, ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz, bw, bh, bd = best_pos
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
            else:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill in any remaining None
        for i in range(n):
            if results[i] is None:
                results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Calculate filled volume
        filled = 0
        for r in results:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        return results, filled
    
    # Strategy 1: Volume descending (BFD)
    results1, filled1 = execute_placement(box_data)
    
    # Strategy 2: Try sorting by max dimension descending, then volume
    box_data_alt = sorted(box_data, key=lambda x: (max(x[3], x[4], x[5]), x[2]), reverse=True)
    results2, filled2 = execute_placement(box_data_alt)
    
    # Strategy 3: Sort by surface area descending, then volume
    box_data_alt2 = sorted(box_data, key=lambda x: (2*(x[3]*x[4] + x[4]*x[5] + x[5]*x[3]), x[2]), reverse=True)
    results3, filled3 = execute_placement(box_data_alt2)
    
    # Choose the best result
    best_results = results1
    best_filled = filled1
    if filled2 > best_filled:
        best_results = results2
        best_filled = filled2
    if filled3 > best_filled:
        best_results = results3
        best_filled = filled3
    
    return best_results

# EVOLVE-BLOCK-END
