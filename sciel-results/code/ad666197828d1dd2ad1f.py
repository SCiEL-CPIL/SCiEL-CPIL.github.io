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
    
    # Use a resolution of 1 unit. If dimensions are large, this might be too coarse,
    # but for typical 60-box instances with container sizes < 100, it's manageable.
    # To be safe, let's cap the grid size to avoid memory explosion.
    # If container is very large, we fall back to the list-based method.
    MAX_GRID_SIZE = 200
    if W > MAX_GRID_SIZE or H > MAX_GRID_SIZE or D > MAX_GRID_SIZE:
        # Fallback to original logic if grid is too large
        return _load_fallback(boxes, container, deadline)

    # Create 3D occupancy grid
    # Shape: (W, H, D)
    # 1 = occupied, 0 = free
    grid = np.zeros((W, H, D), dtype=np.uint8)
    
    def blb_solve(ordered_indices):
        """
        Runs the Bottom-Left-Back heuristic on a specific ordering of box indices.
        Uses the global grid for collision checks.
        Returns the best placement found for this ordering.
        """
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        
        # We need to track which boxes were placed in this specific run to clear the grid
        # But since we are trying multiple strategies, we need to reset the grid for each run.
        # The grid is passed in or reset outside.
        
        # Candidate points: (x, y, z)
        candidates = [(0, 0, 0)]
        
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
                # cx + rw <= W, cy + rh <= H, cz + rd <= D
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check for overlap using grid
                    # Slice the grid for the box volume
                    # x: cx to cx+rw, y: cy to cy+rh, z: cz to cz+rd
                    # Note: NumPy slicing is [x, y, z] if we defined grid that way.
                    # Let's define grid[x, y, z].
                    # Check if any cell in the box volume is occupied (1)
                    # If sum is 0, it's free.
                    box_vol = grid[cx:cx+rw, cy:cy+rh, cz:cz+rd]
                    if np.sum(box_vol) == 0:
                        # Valid position
                        score = cz * 1000000 + cy * 1000 + cx
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                # Record placement
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update grid
                grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                
                # Generate new candidate points
                # 1. On top
                candidates.append((bx, by, bz + bd))
                # 2. To the right
                candidates.append((bx + bw, by, bz))
                # 3. In front
                candidates.append((bx, by + bh, bz))
                # 4. Corner (right and front)
                candidates.append((bx + bw, by + bh, bz))
                # 5. Top-Right
                candidates.append((bx + bw, by, bz + bd))
                # 6. Top-Front
                candidates.append((bx, by + bh, bz + bd))
                # 7. Top-Right-Front
                candidates.append((bx + bw, by + bh, bz + bd))
                
                # Optional: Remove duplicates from candidates to save time?
                # Set is better for duplicates, but list is faster to iterate if small.
                # With 60 boxes, candidates can grow to ~60*7 = 420.
                # Using a set might be better to avoid re-checking same spots.
                # However, converting list to set every time is O(N).
                # Let's stick with list for now, but maybe deduplicate periodically?
                # For now, simple list append.
                
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
    
    # Try each strategy
    for sort_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        
        # Reset grid
        grid[:] = 0
        
        results, completed = blb_solve(sorted_indices)
        
        if completed:
            score = calculate_score(results)
            if score > best_score:
                best_score = score
                best_results = results
    
    # If we have time left, try some random restarts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 20:
            random.shuffle(indices)
            
            # Reset grid
            grid[:] = 0
            
            results, completed = blb_solve(indices)
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
