import time
import itertools
import numpy as np

def _pack_once(boxes, container, sort_key, deadline):
    """Run one packing pass with a given sort key. Returns placements list."""
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by the given key
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: sort_key(x), reverse=True)
    
    placements = [None] * n
    
    # Occupancy grid
    max_grid_cells = 50_000_000
    use_dense_grid = (W * H * D) <= max_grid_cells
    
    if use_dense_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        grid = None

    placed_boxes = []
    
    # Use a set for O(1) deduplication of extreme points
    extreme_points_set = {(0, 0, 0)}
    extreme_points_list = [(0, 0, 0)]
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
        
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        sorted_eps = sorted(extreme_points_list, key=lambda p: (p[2], p[1], p[0]))
        
        for px, py, pz in sorted_eps:
            if time.time() > deadline:
                break
            
            # Track best orientation at this EP by fit-tightness
            local_best = None
            local_best_score = float('inf')
            
            for ow, oh, od in orients:
                if px + ow > W or py + oh > H or pz + od > D:
                    continue
                
                valid = True
                if use_dense_grid:
                    if np.any(grid[px:px+ow, py:py+oh, pz:pz+od]):
                        valid = False
                else:
                    for bx, by, bz, bw, bh, bd in placed_boxes:
                        if (px < bx + bw and bx < px + ow and
                            py < by + bh and by < py + oh and
                            pz < bz + bd and bz < pz + od):
                            valid = False
                            break
                
                if valid:
                    # Score: prefer orientations that are "tighter" (less wasted space along axes)
                    # Penalize large gaps: use (W-ow)*(H-oh)*(D-od) as a proxy for leftover space
                    # Smaller is better
                    tightness = (W - ow) * (H - oh) * (D - od)
                    if tightness < local_best_score:
                        local_best_score = tightness
                        local_best = (px, py, pz, ow, oh, od)
            
            if local_best is not None:
                # Compare with global best using position first, then tightness
                pos_score = local_best[2] * 10000 + local_best[1] * 100 + local_best[0]
                combined = pos_score * 1000000 + local_best_score
                if combined < best_score:
                    best_score = combined
                    best_placement = local_best
                break  # First valid EP in sorted order is best position
        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            placed_boxes.append((px, py, pz, ow, oh, od))
            
            if use_dense_grid:
                grid[px:px+ow, py:py+oh, pz:pz+od] = 1
            
            new_points = [
                (px + ow, py, pz),
                (px, py + oh, pz),
                (px, py, pz + od)
            ]
            
            for p in new_points:
                if p not in extreme_points_set:
                    extreme_points_set.add(p)
                    extreme_points_list.append(p)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements


def _filled_volume(placements, container):
    """Compute total volume of placed boxes."""
    total = 0
    for p in placements:
        if p[0] == 0:
            total += p[4] * p[5] * p[6]
    return total


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 112.0  # 8s margin for safety
    
    # Define multiple sorting strategies
    strategies = [
        lambda x: x[1] * x[2] * x[3],  # Volume descending
        lambda x: max(x[1], x[2], x[3]),  # Max dimension descending
        lambda x: min(x[1], x[2], x[3]),  # Min dimension descending (tight boxes first)
        lambda x: 2 * (x[1]*x[2] + x[2]*x[3] + x[1]*x[3]),  # Surface area descending
        lambda x: x[1] * x[2] * x[3] / max(x[1], x[2], x[3]),  # Volume / max_dim (aspect ratio aware)
    ]
    
    best_placements = None
    best_vol = -1
    
    for si, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        placements = _pack_once(boxes, container, strategy, deadline)
        vol = _filled_volume(placements, container)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    # If we still have time, do additional randomized restarts with shuffled tie-breaks
    if best_placements is not None and time.time() < deadline - 5.0:
        import random
        rng = random.Random(42)
        
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
            
            # Create a slightly perturbed sort: volume with small random jitter
            def jittered_sort(x):
                base = x[1] * x[2] * x[3]
                jitter = rng.random() * 0.01 * base
                return base + jitter
            
            placements = _pack_once(boxes, container, jittered_sort, deadline)
            vol = _filled_volume(placements, container)
            
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    if best_placements is None:
        # Fallback: all unplaced
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
