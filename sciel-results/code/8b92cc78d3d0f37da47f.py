import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0)] * n

    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    box_orientations = [get_orientations(w, h, d) for (w, h, d) in boxes]

    # Grid-based occupancy for fast overlap checks
    # If dimensions are too large for memory, we might need a fallback, 
    # but for typical 60-box problems, dimensions are usually manageable.
    # Let's assume dimensions are integers and fit in memory.
    # If W*H*D is too large, we fall back to list-based, but grid is preferred.
    MAX_GRID_SIZE = 5000000  # 5 million cells
    use_grid = (W * H * D <= MAX_GRID_SIZE)
    
    if use_grid:
        grid = np.zeros((W, H, D), dtype=bool)
    else:
        grid = None
        # Fallback overlap check
        def overlaps_list(placed, x, y, z, w, h, d):
            for _, px, py, pz, pw, ph, pd in placed:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return True
            return False

    def is_valid_placement(x, y, z, w, h, d, placed):
        if x + w > W or y + h > H or z + d > D:
            return False
        
        if use_grid:
            # Check if any part of the box overlaps with occupied cells
            if grid[x:x+w, y:y+h, z:z+d].any():
                return False
        else:
            if overlaps_list(placed, x, y, z, w, h, d):
                return False
        return True

    def mark_placement(x, y, z, w, h, d):
        if use_grid:
            grid[x:x+w, y:y+h, z:z+d] = True

    def unmark_placement(x, y, z, w, h, d):
        if use_grid:
            grid[x:x+w, y:y+h, z:z+d] = False

    def generate_candidates(placed, w, h, d):
        """
        Generate candidate positions (x, y, z) based on the boundaries of placed boxes.
        This is a standard heuristic for 3D packing.
        """
        if not placed:
            return [(0, 0, 0)]
        
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])
        
        for _, px, py, pz, pw, ph, pd in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
            z_cands.add(pz)
            z_cands.add(pz + pd)
            
        # Filter candidates that are out of bounds
        x_cands = {x for x in x_cands if x + w <= W}
        y_cands = {y for y in y_cands if y + h <= H}
        z_cands = {z for z in z_cands if z + d <= D}
        
        # To avoid O(N^3) explosion in candidates, we can limit the number
        # or just iterate. With N=60, unique coords are <= 120 each. 120^3 is ~1.7M.
        # This might be too slow if done for every box.
        # Optimization: Only consider candidates that are "close" to existing boxes?
        # Or simply iterate all combinations. 
        # Let's try to optimize by sorting and limiting if necessary.
        
        candidates = []
        for x in x_cands:
            for y in y_cands:
                for z in z_cands:
                    candidates.append((x, y, z))
                    
        # Sort by a heuristic: prefer lower z, then y, then x (bottom-left-front)
        # This helps in filling from the bottom up.
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        return candidates

    def greedy_pack(sort_key, rng=None, perturbation=0.0):
        """Greedy packing with given sort key."""
        indexed_boxes = sorted(range(n), key=lambda i: sort_key(boxes[i]), reverse=True)
        
        # Add perturbation to sort key to create diversity in restarts
        if perturbation > 0 and rng is not None:
            indexed_boxes = sorted(indexed_boxes, key=lambda i: sort_key(boxes[i]) + rng.uniform(-perturbation, perturbation), reverse=True)

        placed = []  # List of (idx, x, y, z, w, h, d)
        total_vol = 0

        for idx in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            best_placement = None
            best_score = float('inf')

            for ow, oh, od in box_orientations[idx]:
                # Generate candidates for this orientation
                cands = generate_candidates(placed, ow, oh, od)
                
                for cx, cy, cz in cands:
                    if is_valid_placement(cx, cy, cz, ow, oh, od, placed):
                        # Score: prefer lower z, then y, then x
                        score = cz * 10000 + cy * 100 + cx
                        if perturbation > 0 and rng is not None:
                            score += rng.random() * 1.0 # Small random tie-breaker
                        
                        if score < best_score:
                            best_score = score
                            best_placement = (idx, cx, cy, cz, ow, oh, od)
                
                # If we found a valid placement for this orientation, we might want to stop
                # if we found a "good" one (e.g., z=0), but to be safe, we check all orientations
                # and pick the best score.
                if best_placement is not None:
                    # Optimization: If we found a placement at z=0, it's hard to beat.
                    # But let's stick to checking all orientations for correctness.
                    pass

            if best_placement:
                placed.append(best_placement)
                mark_placement(best_placement[1], best_placement[2], best_placement[3], best_placement[4], best_placement[5], best_placement[6])
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol

    # Different sorting strategies
    # 1. Volume
    sort_vol = lambda b: b[0]*b[1]*b[2]
    # 2. Max dimension
    sort_max = lambda b: max(b)
    # 3. Surface area
    sort_surf = lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2])
    # 4. Aspect ratio (closer to 1 is better, so we sort by inverse aspect or similar)
    # Let's use "minimum dimension" as a proxy for "fit in tight spaces" or "aspect"
    sort_min = lambda b: min(b)
    # 5. Base area (w*h)
    sort_base = lambda b: b[0]*b[1]

    strategies = [sort_vol, sort_max, sort_surf, sort_min, sort_base]

    best_result = None
    best_vol = 0

    # Deterministic strategies first
    for si, sort_key in enumerate(strategies):
        if time.time() - start_time > DEADLINE:
            break
        
        # Reset grid if using one
        if use_grid:
            grid[:] = False
        
        res, vol = greedy_pack(sort_key)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    rng = random.Random(42)
    restart_count = 0
    max_restarts = 500 # Can do many more with grid

    while time.time() - start_time < DEADLINE and restart_count < max_restarts:
        restart_count += 1

        # Pick a random strategy
        sort_key = strategies[rng.randint(0, len(strategies) - 1)]
        
        # Add perturbation to create diversity
        perturbation = rng.uniform(0.0, 5.0)
        
        if use_grid:
            grid[:] = False
            
        res, vol = greedy_pack(sort_key, rng, perturbation)
        
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # If no box was placed, return empty
    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n

    return best_result
