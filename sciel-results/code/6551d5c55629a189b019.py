import time
import random
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin for 150s limit
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # Skyline Grid: grid[x, y] = max z-height occupied in column (x, y)
    # We use a 2D array of shape (W, H) initialized to 0.
    # To avoid re-initializing a potentially large W*H array for every restart/iteration,
    # we could use a sparse representation or a flat array with versioning, 
    # but for 60 boxes and typical container sizes, np.zeros is fast enough if we are careful.
    # However, to be safe on speed, we will use a standard list of lists or a flat numpy array.
    # Let's use a flat numpy array for fast access, but we need to reset it.
    # Actually, for local search, re-running the whole pack is the cost.
    # Let's stick to a simple 2D list of ints for the skyline, as W*H might be manageable 
    # and Python list access is fast for small W, H. If W, H are large (e.g., 100x100), 
    # numpy is better. Let's use numpy for the grid but reset it efficiently.
    
    def pack_with_order(order):
        # Initialize skyline grid
        # Using a 2D numpy array for O(1) updates? 
        # Updating a 2D numpy array slice is fast.
        # grid[x, y] = max(grid[x, y], z + dh)
        # This is vectorizable per box placement.
        
        grid = np.zeros((W, H), dtype=np.int32)
        placed = []
        
        # Candidate points: (x, y, z)
        # We maintain a list of candidate "corners" or "skyline points".
        # In the skyline method, valid positions are typically:
        # 1. (0,0,0)
        # 2. For each placed box, its top-face corners and side-face corners.
        # Specifically, if a box is at (x,y,z,w,h,d), new candidate points are:
        # (x+w, y, z), (x, y+h, z), (x, y, z+d) -> these are the "new" boundaries.
        # Also, we should consider points on the existing skyline.
        # A simpler and robust approach for Skyline:
        # Maintain a set of candidate (x, y) coordinates.
        # For each (x, y), the z-coordinate is grid[x, y].
        # Initially candidates = {(0,0)}.
        # When placing a box at (x, y, z, w, h, d):
        #   - Update grid[x:x+w, y:y+h] to be at least z+d.
        #   - Add new candidate (x, y) points? No, we already have them.
        #   - Add new candidate (x+w, y), (x, y+h), (x+w, y+h) if they are within bounds.
        #   - Also, we might need to add points that are "exposed" by the new box?
        #     Actually, the standard 3D Skyline (or "Extreme Points") approach:
        #     A point (x, y, z) is a candidate if it is a corner of a placed box 
        #     and is not "hidden" by another box.
        #     With the grid, we can simply generate candidates from the corners of placed boxes
        #     and check validity using the grid.
        
        # Let's use a list of unique (x, y) pairs as the "footprint" candidates.
        # For each (x, y), the z is determined by grid[x, y].
        # We start with (0, 0).
        # When we place a box at (x, y, z, w, h, d), we add:
        # (x+w, y), (x, y+h), (x+w, y+h) to the candidate set if in bounds.
        # We also keep the existing candidates.
        # This set grows but is bounded by the number of placed boxes * 3.
        
        cand_xys = [(0, 0)]
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            for (ow, oh, od) in orient_cache[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                for (cx, cy) in cand_xys:
                    # Check bounds
                    if cx + ow > W or cy + oh > H:
                        continue
                    
                    # Get current height at this column
                    # We need to check if the box fits on TOP of the skyline.
                    # The box occupies [cx, cx+ow) x [cy, cy+oh).
                    # The required z-start is the MAX of grid[x, y] for all x in [cx, cx+ow), y in [cy, cy+oh).
                    # If we use numpy, we can do:
                    # max_h = np.max(grid[cx:cx+ow, cy:cy+oh])
                    # This is O(ow*oh) but vectorized in C.
                    
                    max_h = np.max(grid[cx:cx+ow, cy:cy+oh])
                    start_z = max_h
                    
                    if start_z + od > D:
                        continue
                    
                    # Valid placement!
                    # Score: prefer lower z, then lower y, then lower x (or similar)
                    # Standard BFDH: minimize z, then y, then x.
                    score = 10000 * start_z + 100 * cy + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, start_z, ow, oh, od)
            
            if best_pos:
                x, y, z, ow, oh, od = best_pos
                placed.append((idx, x, y, z, ow, oh, od))
                
                # Update grid
                # The new height for this region is z + od
                # We only need to update if z + od > current grid value
                # grid[x:x+ow, y:y+oh] = max(grid[x:x+ow, y:y+oh], z + od)
                current_slice = grid[x:x+ow, y:y+oh]
                new_height = z + od
                # np.maximum is element-wise
                grid[x:x+ow, y:y+oh] = np.maximum(current_slice, new_height)
                
                # Update candidates
                # Add the "new" corners
                new_cands = [
                    (x + ow, y),
                    (x, y + oh),
                    (x + ow, y + oh)
                ]
                for nc in new_cands:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H:
                        # Check if this candidate is already in the list
                        # To avoid O(N) search, we can just append and deduplicate later or use a set
                        # Since N=60, the list size is small (< 100), linear search is fine.
                        if nc not in cand_xys:
                            cand_xys.append(nc)
                
                # Optimization: Prune candidates that are no longer "extreme"?
                # A candidate (x, y) is "bad" if there is another candidate (x', y') 
                # such that x' <= x, y' <= y, and grid[x', y'] >= grid[x, y].
                # This is complex. For now, keep all.
                
        return placed

    # --- Search Strategy ---
    
    indexed = list(range(n))
    
    # 1. Generate initial diverse orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    # Aspect ratio (cubic first)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        return max(w, h, d) / min(w, h, d)
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    # Random orders
    random_orders = []
    for _ in range(10):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
        
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    best_order = None
    
    start_time = time.time()
    
    # Phase 1: Greedy BFD with multiple orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_with_order(order)
        v = sum(p[4]*p[5]*p[6] for p in p)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            best_order = order[:]
            if len(p) == n:
                break # Can't do better than 100%

    # Phase 2: Local Search on the Best Order
    if best_order and len(best_placed) < n:
        current_order = best_order[:]
        current_vol = best_vol
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Randomly select two indices to swap
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            p = pack_with_order(new_order)
            v = sum(p[4]*p[5]*p[6] for p in p)
            
            if v > current_vol:
                current_vol = v
                current_order = new_order
                best_vol = v
                best_placed = p[:]
                no_improve_count = 0
                if len(p) == n:
                    break
            else:
                no_improve_count += 1
                
            # Also try 3-opt occasionally
            if random.random() < 0.1:
                i, j, k = random.sample(range(n), 3)
                new_order = current_order[:]
                new_order[i], new_order[j], new_order[k] = new_order[k], new_order[i], new_order[j]
                p = pack_with_order(new_order)
                v = sum(p[4]*p[5]*p[6] for p in p)
                if v > current_vol:
                    current_vol = v
                    current_order = new_order
                    best_vol = v
                    best_placed = p[:]
                    no_improve_count = 0
                    if len(p) == n:
                        break
                else:
                    no_improve_count += 1

    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result
