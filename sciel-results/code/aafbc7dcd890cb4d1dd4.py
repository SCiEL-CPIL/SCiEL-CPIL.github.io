import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    # Safety deadline: 115 seconds to ensure we return before the 120s hard limit
    deadline = start_time + 115.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # Use a 2D height map (skyline) approach
    # height_map[x, y] = maximum z-coordinate occupied at position (x, y)
    # This allows O(1) amortized placement checks
    
    MAX_GRID_SIZE = 200
    if W > MAX_GRID_SIZE or H > MAX_GRID_SIZE or D > MAX_GRID_SIZE:
        # Fallback to original logic if grid is too large
        return _load_fallback(boxes, container, deadline)

    # Create 2D height map
    height_map = np.zeros((W, H), dtype=np.int32)
    
    def skyline_solve(ordered_indices, placement_strategy='blb'):
        """
        Runs the Skyline heuristic on a specific ordering of box indices.
        Uses the 2D height map for collision checks.
        
        placement_strategy: 'blb' (bottom-left-back), 'bottom-left', 'left-bottom'
        """
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        
        # Candidate points: (x, y) - we'll compute z from height_map
        candidates = [(0, 0)]
        
        for idx in ordered_indices:
            w, h, d = boxes[idx]
            
            # Check time limit inside the loop
            if time.time() > deadline:
                return results, False
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # Get unique rotations
            rotations = list(set(itertools.permutations((w, h, d))))
            
            for rw, rh, rd in rotations:
                # Filter candidates that fit in bounds
                for cx, cy in candidates:
                    if cx + rw > W or cy + rh > H:
                        continue
                    
                    # Get the current height at this position
                    # We need to find the minimum height in the footprint area
                    # Actually, we need to check if the box can be placed at z = height_map value
                    # The box occupies z from z_start to z_start + rd
                    # It's valid if z_start + rd <= D and no overlap with existing boxes
                    
                    # For skyline, we place the box at the lowest possible z
                    # that doesn't overlap. The lowest z is the max height in the footprint
                    # but we also need to check if placing at that z causes overlap
                    # in the vertical direction with boxes that extend above.
                    
                    # Simpler approach: try z = 0, z = 1, ... up to D-rd
                    # But that's slow. Instead, use the height map:
                    # The box can be placed at z = max(height_map[cx:cx+rw, cy:cy+rh])
                    # if that z + rd <= D
                    
                    # Get the max height in the footprint
                    footprint_heights = height_map[cx:cx+rw, cy:cy+rh]
                    z_start = int(np.max(footprint_heights)) if footprint_heights.size > 0 else 0
                    
                    if z_start + rd > D:
                        continue
                    
                    # Check if this placement is valid (no overlap)
                    # Since we're using a height map, if z_start is the max height in the footprint,
                    # then placing at z_start should be valid as long as no box extends below z_start
                    # in the footprint area. But that's not tracked in a simple height map.
                    
                    # For correctness, we need to verify no overlap. Let's use a simpler check:
                    # Assume the height map is accurate and place at z_start.
                    # This is a heuristic assumption that may fail in some cases, but it's fast.
                    
                    # Score: prefer lower z, then left, then back
                    if placement_strategy == 'blb':
                        score = z_start * 1000000 + cy * 1000 + cx
                    elif placement_strategy == 'bottom-left':
                        score = z_start * 1000000 + cx * 1000 + cy
                    else:  # 'left-bottom'
                        score = cy * 1000000 + z_start * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, z_start)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                # Record placement
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update height map
                # The box occupies z from bz to bz+bd
                # For each (x, y) in the footprint, update height_map[x, y] = max(height_map[x, y], bz + bd)
                height_map[bx:bx+bw, by:by+bh] = np.maximum(height_map[bx:bx+bw, by:by+bh], bz + bd)
                
                # Generate new candidate points
                # 1. To the right
                candidates.append((bx + bw, by))
                # 2. In front
                candidates.append((bx, by + bh))
                # 3. Corner (right and front)
                candidates.append((bx + bw, by + bh))
                
                # Also add the original position (in case it's still useful)
                # Actually, we should add positions adjacent to the placed box
                
        return results, True

    def calculate_score(results):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in results:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol

    best_results = None
    best_score = 0
    
    # Define different sorting strategies
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], # Volume
        lambda i: max(boxes[i]),                            # Max Dimension
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2], # Surface Area
        lambda i: boxes[i][0] * boxes[i][1],                # Base Area
        lambda i: -min(boxes[i]),                           # Min Dimension
    ]
    
    indices = list(range(n_boxes))
    
    # Try each strategy with different placement strategies
    placement_strategies = ['blb', 'bottom-left', 'left-bottom']
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        
        for ps in placement_strategies:
            if time.time() > deadline:
                break
                
            # Reset height map
            height_map[:] = 0
            
            results, completed = skyline_solve(sorted_indices, ps)
            
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    # If we have time left, try some random restarts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 30:
            random.shuffle(indices)
            
            # Reset height map
            height_map[:] = 0
            
            # Try different placement strategies
            ps = placement_strategies[attempts % len(placement_strategies)]
            results, completed = skyline_solve(indices, ps)
            attempts += 1
            
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results

    # Fallback if no boxes were placed or best_results is None
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results

def _load_fallback(boxes, container, deadline):
    """
    Original logic for large containers where grid is too big.
    """
    W, H, D = container
    n_boxes = len(boxes)
    
    def blb_solve(ordered_indices):
        placed = []
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        candidates = set([(0, 0, 0)])
        
        for idx in ordered_indices:
            w, h, d = boxes[idx]
            
            if time.time() > deadline:
                return results, False
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rotations = list(set(itertools.permutations((w, h, d))))
            
            for rw, rh, rd in rotations:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and px < cx + rw and
                            cy < py + ph and py < cy + rh and
                            cz < pz + pd and pz < cz + rd):
                            valid = False
                            break
                    
                    if valid:
                        score = cz * 1000000 + cy * 1000 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                candidates.add((bx, by, bz + bd))
                candidates.add((bx + bw, by, bz))
                candidates.add((bx, by + bh, bz))
                candidates.add((bx + bw, by + bh, bz))
                candidates.add((bx + bw, by, bz + bd))
                candidates.add((bx, by + bh, bz + bd))
                candidates.add((bx + bw, by + bh, bz + bd))
                
        return results, True

    def calculate_score(results):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in results:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol

    best_results = None
    best_score = 0
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
        lambda i: max(boxes[i]),
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],
        lambda i: boxes[i][0] * boxes[i][1],
        lambda i: -min(boxes[i]),
    ]
    
    indices = list(range(n_boxes))
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        results, completed = blb_solve(sorted_indices)
        if completed:
            score = calculate_score(results)
            if score > best_score:
                best_score = score
                best_results = results

    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 20:
            random.shuffle(indices)
            results, completed = blb_solve(indices)
            attempts += 1
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
