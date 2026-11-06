"""Strong 3D Container Loading: multi-strategy skyline placement with deadline guard."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

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
        
        # Sort orientations: prefer those with smaller max dimension
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    def execute_strategy(box_order):
        """Execute a placement strategy with a specific box ordering."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes)
        
        def get_candidates():
            """Generate candidate positions from placed boxes and origin."""
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
            cx_list, cy_list, cz_list = get_candidates()
            
            # Generate all candidate positions sorted by (z, y, x) to prefer lowest positions
            candidates = []
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        candidates.append((cx, cy, cz))
            candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            
            sorted_orients = sorted(orients, key=lambda o: (max(o), o[0], o[1], o[2]))
            
            for (cx, cy, cz) in candidates:
                for (bw, bh, bd) in sorted_orients:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap
                    if not check_no_overlap(cx, cy, cz, bw, bh, bd):
                        continue
                    
                    # Place the box
                    placed.append((cx, cy, cz, bw, bh, bd))
                    results[idx] = (0, cx, cy, cz, bw, bh, bd)
                    return True
            
            results[idx] = (-1, 0, 0, 0, ow, oh, od)
            return False
        
        # Phase 1: Place boxes in the given order
        for idx, orients, volume, ow, oh, od in box_order:
            if time.time() > deadline:
                break
            try_place_box(idx, orients, volume, ow, oh, od)
        
        # Phase 2: Try to place remaining boxes in ascending volume order
        unplaced = []
        for idx, orients, volume, ow, oh, od in box_order:
            if results[idx] is None or results[idx][0] == -1:
                unplaced.append((idx, orients, volume, ow, oh, od))
        
        unplaced.sort(key=lambda x: x[2])
        
        for idx, orients, volume, ow, oh, od in unplaced:
            if time.time() > deadline:
                break
            try_place_box(idx, orients, volume, ow, oh, od)
        
        # Fill remaining
        for i in range(len(boxes)):
            if results[i] is None:
                w_i, h_i, d_i = boxes[i]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                
        # Calculate score
        filled_vol = 0
        for res in results:
            if res[0] == 0:
                filled_vol += res[4] * res[5] * res[6]
        
        return filled_vol, results
    
    # Define multiple strategies
    strategies = []
    
    # Strategy 1: Volume Descending
    strategies.append(sorted(box_data, key=lambda x: -x[2]))
    
    # Strategy 2: Max Dimension Descending
    strategies.append(sorted(box_data, key=lambda x: -max(x[3], x[4], x[5])))
    
    # Strategy 3: Surface Area Descending
    strategies.append(sorted(box_data, key=lambda x: -(2*(x[3]*x[4] + x[4]*x[5] + x[5]*x[3]))))
    
    # Strategy 4: Volume / Max Dimension (prefer compact large boxes)
    strategies.append(sorted(box_data, key=lambda x: -(x[2] / max(max(x[3], x[4], x[5]), 1))))
    
    # Strategy 5: Product of two largest dimensions (prefer "flat" large boxes)
    def sort_key_flat(x):
        dims = sorted([x[3], x[4], x[5]], reverse=True)
        return -(dims[0] * dims[1])
    strategies.append(sorted(box_data, key=sort_key_flat))
    
    # Strategy 6: Volume with tiebreaker on smallest dimension (prefer thin large boxes)
    strategies.append(sorted(box_data, key=lambda x: (-x[2], min(x[3], x[4], x[5]))))
    
    # Run strategies and keep the best
    best_score = -1
    best_res = None
    
    for i, strategy in enumerate(strategies):
        if time.time() > deadline - 5:
            break
        
        vol, res = execute_strategy(strategy)
        if vol > best_score:
            best_score = vol
            best_res = res
    
    if best_res:
        return best_res
    else:
        # Fallback if something went wrong
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
# EVOLVE-BLOCK-END
