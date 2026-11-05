"""Combined 3D Bin Packing Solution"""
import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []

    # Pre-compute box metrics
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        vol = w * h * d
        max_dim = max(w, h, d)
        surf = 2 * (w * h + h * d + w * d)
        box_data.append((i, w, h, d, vol, max_dim, surf))

    # Define sorting keys
    sort_keys = {
        'vol': lambda x: x[4],
        'maxdim': lambda x: x[5],
        'surf': lambda x: x[6]
    }

    # Define position heuristics
    # Returns a score for a candidate position (lower is better)
    def score_pos_origin(cx, cy, cz, w, h, d):
        return cx + cy + cz

    def score_pos_linear(cx, cy, cz, w, h, d):
        # Prioritize filling the bottom layer (z=0) completely, then next, etc.
        # Then y, then x.
        return cx * H * D + cy * D + cz

    # Define bin selection strategies
    # Returns the index of the best bin from the list of valid (bin_idx, pos) pairs
    # valid_pairs is a list of (bin_idx, pos, bin_vol)
    def select_bin_max_vol(valid_pairs):
        if not valid_pairs:
            return -1, None
        # Pick the bin with the highest current volume
        best_idx = -1
        best_pos = None
        best_vol = -1
        for b_idx, pos, vol in valid_pairs:
            if vol > best_vol:
                best_vol = vol
                best_idx = b_idx
                best_pos = pos
        return best_idx, best_pos

    def select_bin_first(valid_pairs):
        if not valid_pairs:
            return -1, None
        # Pick the first valid bin encountered (lowest index)
        # The list is generated in order of bin index
        return valid_pairs[0][0], valid_pairs[0][1]

    # Core packing function
    def run_strategy(order_key, pos_scoring_func, bin_selection_func):
        indexed_boxes = sorted(box_data, key=sort_keys[order_key], reverse=True)
        
        bins = []       # List of lists of (x, y, z, w, h, d, orig_idx)
        bin_grids = []  # List of numpy boolean arrays
        bin_vols = []   # List of current volumes

        # Pre-allocate grid shape
        grid_shape = (W, H, D)
        
        # Candidate generation helper
        def get_candidates(bin_items):
            if not bin_items:
                return [(0, 0, 0)]
            
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (ox, oy, oz, ow, oh, od, _) in bin_items:
                # Faces
                candidates.add((ox + ow, oy, oz))
                candidates.add((ox, oy + oh, oz))
                candidates.add((ox, oy, oz + od))
                
                # Corners (combinations of faces)
                candidates.add((ox + ow, oy + oh, oz))
                candidates.add((ox + ow, oy, oz + od))
                candidates.add((ox, oy + oh, oz + od))
                candidates.add((ox + ow, oy + oh, oz + od))
            
            # Filter candidates that are out of bounds
            valid_candidates = []
            for cx, cy, cz in candidates:
                if 0 <= cx and cx + 0 <= W: # w,h,d checked later per box
                    valid_candidates.append((cx, cy, cz))
            
            return valid_candidates

        def find_best_position(bin_items, bin_grid, w, h, d):
            """
            Finds the best position for a box in a specific bin using the grid.
            Returns (x, y, z) or None.
            """
            if w > W or h > H or d > D:
                return None

            candidates = get_candidates(bin_items)
            
            best_pos = None
            best_score = float('inf')
            
            for cx, cy, cz in candidates:
                # Bounds check
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                
                # Overlap check using grid
                if np.any(bin_grid[cx:cx+w, cy:cy+h, cz:cz+d]):
                    continue
                
                # Score the position
                score = pos_scoring_func(cx, cy, cz, w, h, d)
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
            
            return best_pos

        # Main loop for this strategy
        for i, w, h, d, vol, _, _ in indexed_boxes:
            valid_pairs = []
            
            # Try to fit in existing bins
            for b_idx, bin_items in enumerate(bins):
                pos = find_best_position(bin_items, bin_grids[b_idx], w, h, d)
                if pos is not None:
                    valid_pairs.append((b_idx, pos, bin_vols[b_idx]))
            
            # Select bin
            best_bin_idx, best_pos = bin_selection_func(valid_pairs)
            
            if best_bin_idx == -1:
                # Start new bin
                new_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d, i)])
                bin_vols.append(vol)
                grid = np.zeros(grid_shape, dtype=bool)
                grid[0:w, 0:h, 0:d] = True
                bin_grids.append(grid)
            else:
                bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, i))
                bin_vols[best_bin_idx] += vol
                grid = bin_grids[best_bin_idx]
                grid[best_pos[0]:best_pos[0]+w, best_pos[1]:best_pos[1]+h, best_pos[2]:best_pos[2]+d] = True

        # Calculate score for this strategy
        # B: number of occupied bins
        B = len(bins)
        
        # Last bin fill
        # "Last bin" is defined by the largest label. Since we append sequentially, 
        # the last bin created has the highest index.
        # However, the problem says "largest label defines the last bin".
        # In our implementation, labels are 0..len(bins)-1. So last bin is bins[-1].
        # But what if we move boxes? We don't move boxes in this simple strategy.
        # So the last bin is the one with the highest index.
        
        if bins:
            last_bin_idx = max(range(len(bins)), key=lambda x: x) # Just len(bins)-1
            last_vol = bin_vols[last_bin_idx]
            container_vol = W * H * D
            fill = last_vol / container_vol if container_vol > 0 else 0.0
            score = B + 0.1 * (1 - fill)
        else:
            score = float('inf')
            
        # Construct placements
        placements = [None] * n
        for b_idx, bin_items in enumerate(bins):
            for item in bin_items:
                x, y, z, bw, bh, bd, orig_idx = item
                placements[orig_idx] = (b_idx, x, y, z, bw, bh, bd)
                
        return score, placements

    # Define the set of strategies to try
    # We want to balance exploration and time.
    # 1. Vol + Origin + MaxVol (Current best)
    # 2. Vol + Linear + First (Layered packing)
    # 3. MaxDim + Origin + First
    # 4. Surf + Linear + MaxVol
    
    strategies = [
        ('vol', score_pos_origin, select_bin_max_vol),
        ('vol', score_pos_linear, select_bin_first),
        ('maxdim', score_pos_origin, select_bin_first),
        ('surf', score_pos_linear, select_bin_max_vol),
    ]

    best_score = float('inf')
    best_placements = None
    
    # Safety deadline: 110 seconds
    deadline = time.time() + 110.0
    
    for order_key, pos_func, bin_func in strategies:
        if time.time() > deadline:
            break
            
        try:
            score, placements = run_strategy(order_key, pos_func, bin_func)
            if score < best_score:
                best_score = score
                best_placements = placements
        except Exception:
            # If a strategy fails, ignore it and try the next
            continue

    # Fallback if all strategies failed or produced no result
    if best_placements is None:
        # Emergency fallback: place each box in its own bin (if they fit individually)
        # This is valid but likely high score
        best_placements = []
        for i, (w, h, d) in enumerate(boxes):
            # Check if box fits in container
            if w <= W and h <= H and d <= D:
                best_placements.append((i, 0, 0, 0, w, h, d))
            else:
                # Box doesn't fit? Problem statement implies they do.
                # But just in case, place at 0,0,0 and hope for the best or crash?
                # The task says "places every box... into numbered bins".
                # If it doesn't fit, it's an invalid input or we must handle it.
                # Assuming inputs are valid.
                best_placements.append((i, 0, 0, 0, w, h, d))
        
        # Ensure list length matches n
        if len(best_placements) < n:
             # This shouldn't happen if inputs are valid
             pass
        # The return format is a list of tuples corresponding to input order.
        # My fallback above appends in order 0..n-1, so it's fine.
        # But the main path returns a list where index i corresponds to box i.
        # The fallback above creates a list of tuples (bin, x, y, z, w, h, d).
        # Wait, the main path returns `placements` which is a list of tuples.
        # The fallback creates a list of tuples.
        # However, the fallback loop `for i, (w, h, d) in enumerate(boxes)` creates
        # tuples `(i, 0, 0, 0, w, h, d)`. Here `i` is the bin index.
        # This matches the format `(bin, x, y, z, w, h, d)`.
        # And the list index corresponds to the box index.
        # So `best_placements[i]` is the placement for `boxes[i]`.
        # This is correct.

    return best_placements
