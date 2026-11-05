def _pack_greedy_regret(boxes, container, deadline):
    """
    Pack boxes using a Greedy heuristic with Regret-based ordering.
    At each step, select the unplaced box with the fewest valid placements.
    """
    W, H, D = container
    n = len(boxes)
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    unplaced = list(range(n))
    
    while unplaced:
        if time.time() > deadline:
            break
            
        # For each unplaced box, find the best placement and count options
        best_box_idx = None
        best_regret = float('inf')
        best_pos_for_box = None
        best_orient_for_box = None
        best_score_for_box = float('inf')
        
        # We might want to prioritize volume if regrets are tied or 0?
        # Let's just pick min regret. If regret is 0, we can't place it, so we should probably skip it or place it if we force it?
        # If regret is 0, it means it doesn't fit anywhere. We should mark it as unplaceable and remove from unplaced?
        # Or, if we remove it, we might lose volume.
        # Standard MCF: if min regret is 0, the problem is infeasible for that box. 
        # In our context, we just leave it out.
        
        # Optimization: If we have many boxes, checking all of them every step is O(N^2).
        # N=60, so 3600 iterations total. Each iteration checks up to 60 boxes.
        # This is fine.
        
        for idx in unplaced:
            w, h, d = boxes[idx]
            options_count = 0
            box_best_score = float('inf')
            box_best_pos = None
            box_best_orient = None
            
            # Generate candidates
            candidates = set([(0, 0, 0)])
            for (px, py, pz, pw, ph, pd) in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                
            for ow, oh, od in _orientations(w, h, d):
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                        options_count += 1
                        # Score for this option (BLF)
                        score = cz * 1000.0 + cy * 100.0 + cx * 1.0
                        if score < box_best_score:
                            box_best_score = score
                            box_best_pos = (cx, cy, cz)
                            box_best_orient = (ow, oh, od)
            
            # Regret is the number of options. We want the box with the FEWEST options (most constrained).
            # But if a box has 0 options, it can't be placed. We should prioritize placing boxes that *can* be placed?
            # If we pick a box with 0 options, we can't place it. We should remove it from unplaced.
            # If we pick a box with >0 options, we place it.
            # Strategy: Find the box with the minimum positive regret.
            # If all regrets are 0, break.
            
            if options_count > 0:
                if options_count < best_regret:
                    best_regret = options_count
                    best_box_idx = idx
                    best_pos_for_box = box_best_pos
                    best_orient_for_box = box_best_orient
                    best_score_for_box = box_best_score
        
        if best_box_idx is None:
            # No box can be placed
            break
            
        # Place the selected box
        idx = best_box_idx
        cx, cy, cz = best_pos_for_box
        ow, oh, od = best_orient_for_box
        placed.append((cx, cy, cz, ow, oh, od))
        placements[idx] = (cx, cy, cz, ow, oh, od)
        unplaced.remove(idx)
        
    return placements, sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in placements)
