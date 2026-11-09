<<<<<<< SEARCH
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # If container is too large for a dense grid, fall back to slower method
    # 200x200 grid is 40k cells, which is fine for NumPy
    MAX_GRID_SIZE = 200
    if W > MAX_GRID_SIZE or H > MAX_GRID_SIZE or D > MAX_GRID_SIZE:
        return _load_fallback(boxes, container, deadline)
    
    def skyline_solve(ordered_indices, placement_strategy='blb'):
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        height_map = np.zeros((W, H), dtype=np.int32)
        
        # Candidate (x, y) positions to try
        # Start with (0, 0)
        candidate_xy = set([(0, 0)])
        placed_count = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False
            
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rotations = list(set(itertools.permutations((w, h, d))))
            
            # We need to try all candidate (x,y) and all rotations
            # To optimize, we can iterate rotations and then candidates
            for rw, rh, rd in rotations:
                # Generate valid candidate points for this rotation size?
                # Or just check all current candidates
                for cx, cy in candidate_xy:
                    if cx + rw > W or cy + rh > H:
                        continue
                    
                    # Get the maximum height in the footprint
                    # This is the z-coordinate where the box would sit
                    footprint = height_map[cx:cx+rw, cy:cy+rh]
                    cz = int(np.max(footprint))
                    
                    if cz + rd > D:
                        continue
                    
                    # If we reach here, the placement is valid
                    # Score it
                    if placement_strategy == 'blb':
                        # Bottom-Left-Bottom: minimize z, then y, then x
                        score = cz * 1000000 + cy * 1000 + cx
                    elif placement_strategy == 'bottom-left':
                        # Bottom-Left: minimize z, then x, then y
                        score = cz * 1000000 + cx * 1000 + cy
                    else:
                        # Left-Bottom: minimize y, then z, then x
                        score = cy * 1000000 + cz * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update height map
                height_map[bx:bx+bw, by:by+bh] = np.maximum(
                    height_map[bx:bx+bw, by:by+bh], bz + bd)
                
                # Generate new candidate (x, y) points
                # New potential bottom-left corners are at the edges of the placed box
                candidate_xy.add((bx + bw, by))
                candidate_xy.add((bx, by + bh))
                candidate_xy.add((bx + bw, by + bh))
                
                placed_count += 1
        
        return results, True
    
    def calculate_score(results):
        return sum(w * h * d for bin_id, x, y, z, w, h, d in results if bin_id == 0)
    
    best_results = None
    best_score = 0
    
    # Sorting strategies
    # 1. Volume (largest first)
    # 2. Max dimension (tallest/thickest first)
    # 3. Surface area
    # 4. Min dimension (to fill gaps?) - usually worse for packing
    # 5. Random
    
    indices = list(range(n_boxes))
    
    sort_strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
        lambda i: max(boxes[i]),
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],
    ]
    
    placement_strategies = ['blb', 'bottom-left', 'left-bottom']
    
    # Deterministic runs
    for sort_func in sort_strategies:
        if time.time() > deadline:
            break
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        for ps in placement_strategies:
            if time.time() > deadline:
                break
            results, completed = skyline_solve(sorted_indices, ps)
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    # Randomized runs
    if best_results is not None and time.time() < deadline - 5.0:
        attempts = 0
        while time.time() < deadline - 2.0 and attempts < 50:
            random.shuffle(indices)
            ps = placement_strategies[attempts % len(placement_strategies)]
            results, completed = skyline_solve(indices, ps)
            attempts += 1
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
=======
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # If container is too large for a dense grid, fall back to slower method
    MAX_GRID_SIZE = 150
    if W > MAX_GRID_SIZE or H > MAX_GRID_SIZE or D > MAX_GRID_SIZE:
        return _load_fallback(boxes, container, deadline)
    
    def skyline_solve(ordered_indices, placement_strategy='blb'):
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        height_map = np.zeros((W, H), dtype=np.int32)
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False
            
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rotations = list(set(itertools.permutations((w, h, d))))
            
            # Full grid scan for each rotation
            for rw, rh, rd in rotations:
                if rw > W or rh > H or rd > D:
                    continue
                
                # Create a sliding window max to find the height at each position
                # For each (cx, cy), we need max of height_map[cx:cx+rw, cy:cy+rh]
                # We can use numpy for this efficiently
                
                # Compute the z-position for all valid (cx, cy) positions
                # Valid range: cx in [0, W-rw], cy in [0, H-rh]
                
                # Use integral image approach or direct computation
                # For small grids, direct is fine; for larger, we optimize
                
                if W - rw <= 0 or H - rh <= 0:
                    continue
                
                # Create the sub-arrays for each position
                # This is O((W-rw+1)*(H-rh+1)*rw*rh) which can be expensive
                # Better: use scipy.ndimage or manual convolution
                
                # Simple approach: iterate over valid positions
                for cx in range(0, W - rw + 1):
                    for cy in range(0, H - rh + 1):
                        footprint = height_map[cx:cx+rw, cy:cy+rh]
                        cz = int(np.max(footprint))
                        
                        if cz + rd > D:
                            continue
                        
                        # Score the placement
                        if placement_strategy == 'blb':
                            score = cz * 1000000 + cy * 1000 + cx
                        elif placement_strategy == 'bottom-left':
                            score = cz * 1000000 + cx * 1000 + cy
                        elif placement_strategy == 'compact':
                            # Minimize distance to origin
                            score = cx * cx + cy * cy + cz * cz
                        else:
                            score = cy * 1000000 + cz * 1000 + cx
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update height map
                height_map[bx:bx+bw, by:by+bh] = np.maximum(
                    height_map[bx:bx+bw, by:by+bh], bz + bd)
        
        return results, True
    
    def calculate_score(results):
        return sum(w * h * d for bin_id, x, y, z, w, h, d in results if bin_id == 0)
    
    best_results = None
    best_score = 0
    
    indices = list(range(n_boxes))
    
    # More diverse sorting strategies
    sort_strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i]),  # Max dimension
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],  # Surface area
        lambda i: min(boxes[i]),  # Min dimension
        lambda i: max(boxes[i]) / min(boxes[i]) + 1e-9,  # Aspect ratio (high first)
        lambda i: -max(boxes[i]) / min(boxes[i]) - 1e-9,  # Aspect ratio (low first)
    ]
    
    placement_strategies = ['blb', 'bottom-left', 'compact', 'left-bottom']
    
    # Deterministic runs
    for sort_func in sort_strategies:
        if time.time() > deadline:
            break
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        for ps in placement_strategies:
            if time.time() > deadline:
                break
            results, completed = skyline_solve(sorted_indices, ps)
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    # Randomized runs with more attempts
    if best_results is not None and time.time() < deadline - 5.0:
        attempts = 0
        while time.time() < deadline - 2.0 and attempts < 100:
            # Use different random orderings
            random.shuffle(indices)
            ps = placement_strategies[attempts % len(placement_strategies)]
            results, completed = skyline_solve(indices, ps)
            attempts += 1
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
>>>>>>> REPLACE

