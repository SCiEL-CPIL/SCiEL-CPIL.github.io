"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Grid resolution: aim for ~1000-5000 cells total for balance of speed/accuracy
    # Max dimension is usually the constraining factor for grid size
    max_dim = max(W, H, D)
    if max_dim == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Target ~2000 cells total volume, so side length ~ 12
    target_cells = 2000
    side_len = int(math.pow(target_cells, 1.0/3.0))
    if side_len < 10: side_len = 10
    if side_len > 50: side_len = 50
    
    grid_w = max(1, int(math.ceil(W / max_dim * side_len)))
    grid_h = max(1, int(math.ceil(H / max_dim * side_len)))
    grid_d = max(1, int(math.ceil(D / max_dim * side_len)))
    
    # Ensure grid isn't too huge
    if grid_w * grid_h * grid_d > 50000:
        # Reduce resolution
        scale = math.pow(50000.0 / (grid_w * grid_h * grid_d), 1.0/3.0)
        grid_w = max(1, int(grid_w * scale))
        grid_h = max(1, int(grid_h * scale))
        grid_d = max(1, int(grid_d * scale))

    cell_w = W / grid_w
    cell_h = H / grid_h
    cell_d = D / grid_d

    def get_orientations(w, h, d):
        """Get unique orientations of a box."""
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        orientations = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orientations.append(perm)
        return orientations

    def run_strategy(order_key_func, use_gap_fill=True):
        """
        Runs a packing strategy using grid-based collision detection.
        """
        if time.time() > deadline:
            return 0, None

        indexed_boxes = list(enumerate(boxes))
        try:
            indexed_boxes.sort(key=order_key_func)
        except Exception:
            return 0, None

        # Grid: 3D list of booleans
        # Flattened 1D list for speed: index = z * grid_h * grid_w + y * grid_w + x
        grid_size = grid_w * grid_h * grid_d
        grid = [0] * grid_size
        
        placed = []  # List of (x, y, z, w, h, d, orig_idx)
        results = [None] * len(boxes)
        total_vol = 0

        def is_free(x, y, z, bw, bh, bd):
            """Check if box fits in grid without collision and within bounds."""
            if x + bw > W or y + bh > H or z + bd > D:
                return False
            
            # Convert to grid coordinates
            gx1 = int(x / cell_w)
            gy1 = int(y / cell_h)
            gz1 = int(z / cell_d)
            
            gx2 = int((x + bw) / cell_w)
            gy2 = int((y + bh) / cell_h)
            gz2 = int((z + bd) / cell_d)
            
            # Clamp to grid bounds
            gx1 = max(0, min(grid_w - 1, gx1))
            gy1 = max(0, min(grid_h - 1, gy1))
            gz1 = max(0, min(grid_d - 1, gz1))
            gx2 = max(0, min(grid_w, gx2))
            gy2 = max(0, min(grid_h, gy2))
            gz2 = max(0, min(grid_d, gz2))

            # Check cells
            # Optimization: Only check if any cell is occupied
            for gz in range(gz1, gz2):
                base_z = gz * grid_h * grid_w
                for gy in range(gy1, gy2):
                    base_y = base_z + gy * grid_w
                    for gx in range(gx1, gx2):
                        if grid[base_y + gx]:
                            return False
            return True

        def place_box(x, y, z, bw, bh, bd, orig_idx):
            """Mark box as occupied in grid and record placement."""
            gx1 = int(x / cell_w)
            gy1 = int(y / cell_h)
            gz1 = int(z / cell_d)
            
            gx2 = int((x + bw) / cell_w)
            gy2 = int((y + bh) / cell_h)
            gz2 = int((z + bd) / cell_d)
            
            gx1 = max(0, min(grid_w - 1, gx1))
            gy1 = max(0, min(grid_h - 1, gy1))
            gz1 = max(0, min(grid_d - 1, gz1))
            gx2 = max(0, min(grid_w, gx2))
            gy2 = max(0, min(grid_h, gy2))
            gz2 = max(0, min(grid_d, gz2))

            for gz in range(gz1, gz2):
                base_z = gz * grid_h * grid_w
                for gy in range(gy1, gy2):
                    base_y = base_z + gy * grid_w
                    for gx in range(gx1, gx2):
                        grid[base_y + gx] = 1
            
            placed.append((x, y, z, bw, bh, bd, orig_idx))
            results[orig_idx] = (0, x, y, z, bw, bh, bd)

        def find_spot(bw, bh, bd):
            """
            Find the best spot for a box.
            Strategy: Iterate through potential positions.
            To keep it fast, we use a "shelf" approach or random sampling + local refinement.
            For simplicity and speed, we'll use a hybrid:
            1. Try (0,0,0)
            2. Try positions aligned with existing box edges (corners) but limited set
            3. Random sampling if not found
            """
            # 1. Try origin
            if is_free(0, 0, 0, bw, bh, bd):
                return (0, 0, 0)
            
            # 2. Try candidate points from existing boxes (limited to avoid O(N^2) explosion in inner loop)
            # We maintain a set of candidate points. To keep it simple and fast,
            # we'll just check a few key points: 
            # - Above each placed box
            # - Right of each placed box
            # - Front of each placed box
            # But with 60 boxes, N^2 is 3600 checks. Each check is O(1) grid. 3600 is fine.
            
            candidates = set()
            candidates.add((0,0,0))
            for px, py, pz, pw, ph, pd, _ in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                
            # Sort candidates by "desirability" (lowest x, then y, then z) to pack tight
            # This is a heuristic.
            for cx, cy, cz in sorted(candidates):
                if is_free(cx, cy, cz, bw, bh, bd):
                    return (cx, cy, cz)
            
            # 3. Random search if not found (rare if grid is fine enough and we check corners)
            # Limit random attempts to avoid timeout
            for _ in range(50):
                rx = random.uniform(0, W - bw)
                ry = random.uniform(0, H - bh)
                rz = random.uniform(0, D - bd)
                if is_free(rx, ry, rz, bw, bh, bd):
                    return (rx, ry, rz)
            
            return None

        # Main packing loop
        for orig_idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break

            orientations = get_orientations(w, h, d)
            placed_this = False
            
            # Try orientations. Sort by "compactness" (minimize max dimension or similar)
            # Or just try all.
            for ow, oh, od in orientations:
                spot = find_spot(ow, oh, od)
                if spot is not None:
                    x, y, z = spot
                    place_box(x, y, z, ow, oh, od, orig_idx)
                    total_vol += ow * oh * od
                    placed_this = True
                    break
            
            if not placed_this:
                results[orig_idx] = (-1, 0, 0, 0, w, h, d)

        # Gap filling phase: try to insert left-out boxes
        if use_gap_fill and time.time() < deadline - 5:
            # Sort left-out boxes by volume ascending (smaller first)
            left_out = [i for i in range(len(boxes)) if results[i][0] == -1]
            left_out.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
            
            for i in left_out:
                if time.time() > deadline - 2:
                    break
                w, h, d = boxes[i]
                orientations = get_orientations(w, h, d)
                placed_this = False
                for ow, oh, od in orientations:
                    spot = find_spot(ow, oh, od)
                    if spot is not None:
                        x, y, z = spot
                        place_box(x, y, z, ow, oh, od, i)
                        total_vol += ow * oh * od
                        placed_this = True
                        break
                if not placed_this:
                    # Ensure it's marked as left out (it already is)
                    pass

        # Fill any remaining results
        for i in range(len(boxes)):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        return total_vol, results

    # Define multiple sorting strategies
    def sort_vol_desc(x):
        w, h, d = x[1]
        return -(w * h * d)

    def sort_height_desc(x):
        w, h, d = x[1]
        return -max(w, h, d)

    def sort_sum_desc(x):
        w, h, d = x[1]
        return -(w + h + d)

    def sort_aspect(x):
        w, h, d = x[1]
        vol = w * h * d
        if vol == 0:
            return 0
        s = (w + h + d) / 3.0
        dev = abs(w - s) + abs(h - s) + abs(d - s)
        return (-vol, dev)
    
    def sort_vol_aspect(x):
        w, h, d = x[1]
        vol = w * h * d
        aspect = max(w,h,d) / min(w,h,d) if min(w,h,d)>0 else 0
        return (-vol, aspect)

    best_vol = 0
    best_results = None

    strategies = [
        sort_vol_desc,
        sort_height_desc,
        sort_sum_desc,
        sort_aspect,
        sort_vol_aspect
    ]

    # Run multiple times with different random seeds for randomization in find_spot
    # and potentially different orderings if we add shuffling
    # For now, just run the fixed orderings. The randomization in find_spot (step 3) helps.
    
    for strat in strategies:
        if time.time() > deadline:
            break
        vol, res = run_strategy(strat)
        if res is not None and vol > best_vol:
            best_vol = vol
            best_results = res

    # If we have time, do a few random restarts with shuffled order
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Estimate time per run
        # Just do a few random shuffles
        for _ in range(3): # Max 3 random restarts
            if time.time() > deadline - 5:
                break
            # Shuffle the boxes list copy for this run? 
            # run_strategy takes order_key_func. 
            # We can create a key func that uses a random permutation index.
            perm = list(range(len(boxes)))
            random.shuffle(perm)
            def random_order(x, perm=perm):
                return perm[x[0]]
            
            vol, res = run_strategy(random_order)
            if res is not None and vol > best_vol:
                best_vol = vol
                best_results = res

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
# EVOLVE-BLOCK-END
