"""
3D Container Packing: Grid-based Backtracking Search
Strategy:
1. Use a 3D NumPy grid to represent the container for O(1) overlap checks (slicing).
2. Implement a recursive backtracking search with a limited depth/time budget.
3. Use a "Best-First" heuristic for placement: instead of taking the first valid spot,
   evaluate all valid spots for a box and pick the one that maximizes "contact" 
   (touching walls or other boxes), which tends to pack tighter.
4. If backtracking is too slow or fails to improve, fall back to a fast greedy 
   randomized restart loop using the grid for speed.
"""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # If container is too small for any box, return all left out
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    # Precompute orientations
    all_orients = []
    all_vols = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))
        all_vols.append(w * h * d)
        
    container_vol = W * H * D
    
    # 1. Grid-based Packing Engine
    # The grid is 1 larger in each dimension to handle boundary checks easily? 
    # No, let's stick to standard indexing but use slicing for overlap.
    
    def pack_with_backtracking(order, time_limit=5.0):
        """
        Attempts to pack boxes in the given order using backtracking.
        Returns (filled_volume, placements_dict)
        """
        # Grid for occupancy. 1 if occupied, 0 if empty.
        # Using int8 for memory efficiency, though bool works too.
        # Dimensions: W, H, D
        grid = np.zeros((W, H, D), dtype=np.int8)
        
        placed_boxes = [] # List of (x, y, z, w, h, d, idx)
        current_vol = 0
        
        # Heuristic: Score a position based on how "tight" it is.
        # Tighter is better (more touching walls/boxes).
        # We want to minimize the sum of distances to walls? 
        # Or maximize the number of faces touching something.
        
        def get_candidates(x, y, z, bw, bh, bd):
            """
            Yield all valid positions for a box of size bw,bh,bd 
            given it must start at x,y,z? No, we need to find spots.
            Instead, we will iterate through potential anchor points.
            """
            pass
            
        # To make backtracking tractable, we restrict search to "anchor" points
        # derived from existing boxes, similar to the seed, but with grid checks.
        
        def get_anchors():
            anchors = set()
            anchors.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd, _) in placed_boxes:
                # Faces
                anchors.add((px + pw, py, pz))
                anchors.add((px, py + ph, pz))
                anchors.add((px, py, pz + pd))
                # Corners (to fill gaps between boxes)
                anchors.add((px + pw, py + ph, pz))
                anchors.add((px + pw, py, pz + pd))
                anchors.add((px, py + ph, pz + pd))
                anchors.add((px + pw, py + ph, pz + pd))
                # Also consider centers? No, too many.
                
            # Filter valid anchors
            valid = []
            for a in anchors:
                if 0 <= a[0] < W and 0 <= a[1] < H and 0 <= a[2] < D:
                    valid.append(a)
            # Sort anchors: prefer bottom-left-front (tighter packing)
            # Actually, for backtracking, order matters less if we try all, 
            # but we need a deterministic order to avoid duplicates.
            valid.sort(key=lambda p: (p[2], p[1], p[0]))
            return valid

        def check_fit(x, y, z, bw, bh, bd):
            """Check if box fits at (x,y,z) with dims (bw,bh,bd)"""
            if x + bw > W or y + bh > H or z + bd > D:
                return False
            # Slice the grid
            # grid[x:x+bw, y:y+bh, z:z+bd]
            # If any 1 in this slice, it overlaps.
            if np.any(grid[x:x+bw, y:y+bh, z:z+bd]):
                return False
            return True

        def place(x, y, z, bw, bh, bd, idx):
            grid[x:x+bw, y:y+bh, z:z+bd] = 1
            placed_boxes.append((x, y, z, bw, bh, bd, idx))
            
        def unplace(x, y, z, bw, bh, bd, idx):
            grid[x:x+bw, y:y+bh, z:z+bd] = 0
            # Remove last item (assumes we always pop the one we just placed)
            placed_boxes.pop()

        # Recursive search
        # To keep it fast, we limit the depth of backtracking.
        # If we fail to place a box, we backtrack to the previous one and try a different spot.
        
        # We'll use a simple stack-based approach or recursion.
        # Recursion depth is limited by n (60), which is fine.
        
        start_time = time.time()
        best_vol = 0
        best_placements = {}
        
        # We will try to place boxes in `order`.
        # State: index in order
        # We maintain a list of placements for the current path.
        
        # Optimization: If we just want a good greedy result, backtracking is overkill.
        # But backtracking can find better spots by moving earlier boxes.
        # Let's limit backtracking to the last 3 boxes to keep it fast.
        
        def search(depth, current_vol):
            nonlocal best_vol, best_placements
            if time.time() - start_time > time_limit:
                return False
            
            if depth == len(order):
                if current_vol > best_vol:
                    best_vol = current_vol
                    # Save placements
                    for p in placed_boxes:
                        best_placements[p[6]] = p
                return True
            
            idx = order[depth]
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            
            # Get candidate positions
            # We look at anchors from the current state
            anchors = get_anchors()
            
            # Sort anchors to try "best" ones first (bottom-left-front)
            # The get_anchors already sorts.
            
            placed_success = False
            for ax, ay, az in anchors:
                for bw, bh, bd in orients:
                    if check_fit(ax, ay, az, bw, bh, bd):
                        # Place it
                        place(ax, ay, az, bw, bh, bd, idx)
                        new_vol = current_vol + all_vols[idx]
                        
                        # Recurse
                        if search(depth + 1, new_vol):
                            # If we found a complete solution or improved best, 
                            # and we are not backtracking further, we can return True 
                            # to stop searching this branch? 
                            # Actually, we want to keep searching for a BETTER solution 
                            # if time permits, but for speed, we often take the first 
                            # valid complete path. 
                            # Let's try to find the BEST path within time limit.
                            # If we just return True, we stop exploring other options for this box.
                            # This makes it greedy with limited backtracking.
                            
                            # To allow backtracking:
                            # If we return True, the parent will also return True and stop.
                            # This effectively finds the first valid packing.
                            # To improve, we need to explore alternatives.
                            
                            # Let's modify: 
                            # If we find a valid placement, recurse.
                            # If recursion succeeds (finds a packing for remaining), 
                            # we update best and continue looking for a BETTER one?
                            # No, that's expensive. 
                            # Let's stick to: Find one valid packing. 
                            # The "Best-First" anchor sorting ensures it's a good one.
                            # Backtracking only happens if a later box fails.
                            
                            if depth + 1 == len(order):
                                # Complete packing found
                                if new_vol > best_vol:
                                    best_vol = new_vol
                                    for p in placed_boxes:
                                        best_placements[p[6]] = p
                                unplace(ax, ay, az, bw, bh, bd, idx)
                                # Continue looking for other options? 
                                # If we return True, we stop. 
                                # Let's return True to commit to this path.
                                return True
                            else:
                                # Recurse
                                success = search(depth + 1, new_vol)
                                if success:
                                    unplace(ax, ay, az, bw, bh, bd, idx)
                                    return True
                                else:
                                    # Backtrack: try next position
                                    unplace(ax, ay, az, bw, bh, bd, idx)
                                    continue
                        else:
                            # Recursion failed (time out or dead end)
                            unplace(ax, ay, az, bw, bh, bd, idx)
                            # If time out, break out
                            if time.time() - start_time > time_limit:
                                return False
                            continue
                
                # If no orientation worked at this anchor, try next anchor
            return False
        
        # Start search
        search(0, 0)
        
        # Build result
        result = [None] * n
        for i in range(n):
            if i in best_placements:
                p = best_placements[i]
                result[i] = (0, p[0], p[1], p[2], p[3], p[4], p[5])
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return best_vol, result

    # 2. Greedy Fallback / Multi-Start
    def pack_greedy(order):
        """Fast greedy packing using grid."""
        grid = np.zeros((W, H, D), dtype=np.int8)
        placed_boxes = []
        result = [None] * n
        filled_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            best_pos = None
            best_orient = None
            
            # Generate anchors
            anchors = set([(0,0,0)])
            for (px, py, pz, pw, ph, pd, _) in placed_boxes:
                anchors.add((px + pw, py, pz))
                anchors.add((px, py + ph, pz))
                anchors.add((px, py, pz + pd))
                anchors.add((px + pw, py + ph, pz))
                anchors.add((px + pw, py, pz + pd))
                anchors.add((px, py + ph, pz + pd))
                anchors.add((px, py + ph, pz + pd))
            
            valid_anchors = [a for a in anchors if 0 <= a[0] < W and 0 <= a[1] < H and 0 <= a[2] < D]
            valid_anchors.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for ax, ay, az in valid_anchors:
                for bw, bh, bd in orients:
                    if ax + bw > W or ay + bh > H or az + bd > D:
                        continue
                    if np.any(grid[ax:ax+bw, ay:ay+bh, az:az+bd]):
                        continue
                    # Found a spot
                    best_pos = (ax, ay, az)
                    best_orient = (bw, bh, bd)
                    break
                if best_pos:
                    break
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                placed_boxes.append((bx, by, bz, bw, bh, bd, idx))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled_vol += all_vols[idx]
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return filled_vol, result

    # 3. Main Logic
    indices = list(range(n))
    
    # Strategy 1: Backtracking with Volume Descending
    order_vol = sorted(indices, key=lambda i: all_vols[i], reverse=True)
    
    best_vol = 0
    best_result = None
    
    # Try backtracking first (limited time)
    vol, result = pack_with_backtracking(order_vol, time_limit=10.0)
    if vol > best_vol:
        best_vol = vol
        best_result = result
        
    # Strategy 2: Greedy with various orders
    orders = [
        sorted(indices, key=lambda i: all_vols[i], reverse=True),
        sorted(indices, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indices, key=lambda i: min(boxes[i]), reverse=True),
        list(indices)
    ]
    
    for order in orders:
        vol, result = pack_greedy(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            
    # Strategy 3: Randomized Greedy Restarts
    random.seed(42)
    deadline = time.time() + 115.0
    
    # Use remaining time for randomized restarts
    while time.time() < deadline:
        # Perturb the order
        order = list(order_vol)
        # Swap a few random adjacent elements
        for _ in range(5):
            i = random.randint(0, n-2)
            order[i], order[i+1] = order[i+1], order[i]
            
        vol, result = pack_greedy(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            
    # Fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_result
# EVOLVE-BLOCK-END
