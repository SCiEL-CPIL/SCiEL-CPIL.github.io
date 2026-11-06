"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute all 6 orientations for each box
    # Orientation: (w, h, d)
    orientations = []
    for w, h, d in boxes:
        ors = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
        # Remove duplicates
        seen = set()
        unique = []
        for o in ors:
            if o not in seen:
                seen.add(o)
                unique.append(o)
        orientations.append(unique)
    
    # Sort boxes by volume descending, then by max dimension descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]), 
                                     -max(boxes[i][0], boxes[i][1], boxes[i][2])))
    
    # Greedy placement with multiple strategies
    # We'll try a few heuristics and pick the best
    
    def try_placement(order, orientation_strategy):
        """
        Try placing boxes in given order with given orientation strategy.
        Returns list of placements and total volume placed.
        """
        # Use a simple grid-based or list-based overlap check
        # For efficiency with 60 boxes, we can use a list of placed boxes and check overlap
        
        placed = []  # list of (x, y, z, w, h, d)
        result = [-1] * n  # index -> box index, -1 means not placed
        total_vol = 0
        
        # For each box in order, try to place it
        for idx in order:
            w_orig, h_orig, d_orig = boxes[idx]
            ors = orientations[idx]
            
            # Choose orientation based on strategy
            if orientation_strategy == 'best_fit_x':
                # Try all orientations, pick the one that fits with smallest x
                best_pos = None
                best_orient = None
                for ow, oh, od in ors:
                    # Try to find a position
                    # Simple approach: try to place at lowest possible z, then y, then x
                    # We'll do a simple scan
                    pos = find_position(placed, ow, oh, od, W, H, D)
                    if pos is not None:
                        x, y, z = pos
                        if best_pos is None or (z, y, x) < (best_pos[2], best_pos[1], best_pos[0]):
                            best_pos = pos
                            best_orient = (ow, oh, od)
            elif orientation_strategy == 'min_max_dim':
                # Pick orientation that minimizes the max dimension
                ors_sorted = sorted(ors, key=lambda o: max(o))
                for ow, oh, od in ors_sorted:
                    pos = find_position(placed, ow, oh, od, W, H, D)
                    if pos is not None:
                        best_pos = pos
                        best_orient = (ow, oh, od)
                        break
                if best_pos is None:
                    # Try all orientations
                    for ow, oh, od in ors:
                        pos = find_position(placed, ow, oh, od, W, H, D)
                        if pos is not None:
                            best_pos = pos
                            best_orient = (ow, oh, od)
                            break
            else:  # 'first_fit'
                for ow, oh, od in ors:
                    pos = find_position(placed, ow, oh, od, W, H, D)
                    if pos is not None:
                        best_pos = pos
                        best_orient = (ow, oh, od)
                        break
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed.append((x, y, z, ow, oh, od))
                result[idx] = (0, x, y, z, ow, oh, od)
                total_vol += ow * oh * od
            else:
                result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        return result, total_vol
    
    def find_position(placed, w, h, d, W, H, D):
        """
        Find a position (x, y, z) where a box of size (w, h, d) fits.
        Returns (x, y, z) or None.
        Uses a simple heuristic: try positions based on existing boxes.
        """
        # Candidate positions: origin, and edges of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Positions adjacent to existing boxes
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Also try some offsets
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Also add some grid positions
        # Try to be efficient: limit candidates
        best_pos = None
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check overlap with all placed boxes
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed:
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    overlap = True
                    break
            if not overlap:
                if best_pos is None or (z, y, x) < (best_pos[2], best_pos[1], best_pos[0]):
                    best_pos = (x, y, z)
        
        return best_pos
    
    # Try different orderings and strategies
    best_result = None
    best_vol = 0
    
    # Strategy 1: Volume descending, best_fit_x
    result1, vol1 = try_placement(box_indices, 'best_fit_x')
    if vol1 > best_vol:
        best_vol = vol1
        best_result = result1
    
    # Strategy 2: Volume descending, min_max_dim
    result2, vol2 = try_placement(box_indices, 'min_max_dim')
    if vol2 > best_vol:
        best_vol = vol2
        best_result = result2
    
    # Strategy 3: Volume descending, first_fit
    result3, vol3 = try_placement(box_indices, 'first_fit')
    if vol3 > best_vol:
        best_vol = vol3
        best_result = result3
    
    # Strategy 4: Try sorting by different criteria
    # Sort by max dimension descending
    order4 = sorted(range(n), key=lambda i: -max(boxes[i]))
    result4, vol4 = try_placement(order4, 'best_fit_x')
    if vol4 > best_vol:
        best_vol = vol4
        best_result = result4
    
    # Strategy 5: Sort by sum of dimensions descending
    order5 = sorted(range(n), key=lambda i: -(sum(boxes[i])))
    result5, vol5 = try_placement(order5, 'min_max_dim')
    if vol5 > best_vol:
        best_vol = vol5
        best_result = result5
    
    # Strategy 6: Randomized attempts (limited)
    import random
    random.seed(42)
    for attempt in range(5):
        order = list(range(n))
        random.shuffle(order)
        # Bias towards larger boxes first
        order.sort(key=lambda i: (-boxes[i][0] * boxes[i][1] * boxes[i][2], random.random()))
        strat = random.choice(['best_fit_x', 'min_max_dim', 'first_fit'])
        result_r, vol_r = try_placement(order, strat)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r
    
    if best_result is None:
        # Fallback: place nothing
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result
# EVOLVE-BLOCK-END
