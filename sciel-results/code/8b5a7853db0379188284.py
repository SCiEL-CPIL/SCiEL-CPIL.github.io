import time
import random
import itertools
import numpy as np

def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _try_place_eap_fast(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """
    Fast EAP with sorted candidates and early termination.
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = [None] * n
    candidates = [(0, 0, 0)]
    
    # Pre-compute rotations
    box_rots = [_rotations(*b) for b in boxes]
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[orig_idx]
        rots = box_rots[orig_idx]
        
        # Sort candidates by scoring priority
        sorted_candidates = sorted(candidates, key=lambda p: scoring_func(p, (1,1,1)))
        
        best_pos = None
        best_dim = None
        
        for (cx, cy, cz) in sorted_candidates:
            found = False
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                best_pos = (cx, cy, cz)
                best_dim = (rw, rh, rd)
                found = True
                break
            if found:
                break
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_dim
            placed.append((x, y, z, rw, rh, rd))
            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
            
            # Generate new candidates
            new_pts = [
                (x + rw, y, z),
                (x, y + rh, z),
                (x, y, z + rd),
                (x + rw, y + rh, z),
                (x + rw, y, z + rd),
                (x, y + rh, z + rd),
                (x + rw, y + rh, z + rd)
            ]
            for p in new_pts:
                if 0 <= p[0] <= W and 0 <= p[1] <= H and 0 <= p[2] <= D:
                    if p not in candidates:
                        candidates.append(p)
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _local_search_refine(placements, boxes, container, deadline, max_iters=50):
    """
    Try to improve packing by moving or swapping boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    # Extract placed boxes
    placed_info = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            placed_info.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
    
    if not placed_info:
        return placements, sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)
    
    # Build list of placed boxes for overlap checking
    def get_placed_list():
        return [(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6]) for pi in placed_info]
    
    best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)
    best_placements = [list(p) for p in placements]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try moving each placed box to a better position
        improved = False
        for idx in range(len(placed_info)):
            if time.time() > deadline:
                break
            
            orig_i, x, y, z, w, h, d = placed_info[idx]
            rots = _rotations(w, h, d)
            
            # Remove this box from placed list temporarily
            temp_placed = [pi for j, pi in enumerate(placed_info) if j != idx]
            
            # Generate candidate positions
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in temp_placed:
                points = [
                    (px + pw, py, pz),
                    (px, py + ph, pz),
                    (px, py, pz + pd),
                    (px + pw, py + ph, pz),
                    (px + pw, py, pz + pd),
                    (px, py + ph, pz + pd),
                    (px + pw, py + ph, pz + pd),
                ]
                for pt in points:
                    if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                        if pt not in candidates:
                            candidates.append(pt)
            
            # Try to find a better position (lower z, then y, then x)
            current_score = (z, y, x)
            best_new_pos = None
            best_new_dim = None
            best_new_score = None
            
            for (cx, cy, cz) in sorted(candidates, key=lambda p: (p[2], p[1], p[0])):
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if not _no_overlap((rw, rh, rd), (cx, cy, cz), temp_placed):
                        continue
                    
                    new_score = (cz, cy, cx)
                    if best_new_score is None or new_score < best_new_score:
                        best_new_score = new_score
                        best_new_pos = (cx, cy, cz)
                        best_new_dim = (rw, rh, rd)
                
                if best_new_pos is not None:
                    # Check if this is actually better than current
                    if best_new_score < current_score:
                        break
            
            if best_new_pos is not None and best_new_score < current_score:
                # Update the placement
                nx, ny, nz = best_new_pos
                rw, rh, rd = best_new_dim
                placed_info[idx] = (orig_i, nx, ny, nz, rw, rh, rd)
                placements[orig_i] = (0, nx, ny, nz, rw, rh, rd)
                improved = True
        
        if not improved:
            break
    
    # Rebuild final placements
    final_placements = [list(p) for p in best_placements]
    for pi in placed_info:
        orig_i = pi[0]
        final_placements[orig_i] = (0, pi[1], pi[2], pi[3], pi[4], pi[5], pi[6])
    
    volume = sum(p[4]*p[5]*p[6] for p in final_placements if p[0]==0)
    return final_placements, volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    strategies = []
    
    # Standard Heuristics (Fast EAP)
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('eap', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('eap', mixed1, None, None))
    
    # Custom Scoring Functions
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    
    # Randomized Ties with Standard Score
    rngs = [random.Random(i) for i in range(10)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
        
    # Random Orders
    rng_main = random.Random(42)
    for _ in range(10):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))
    
    best_placements = None
    best_volume = -1
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline:
            break
            
        placements, vol = _try_place_eap_fast(boxes, container, order, deadline, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    # Local search refinement on the best solution
    if best_placements is not None and time.time() < deadline - 5:
        refined_placements, refined_vol = _local_search_refine(
            best_placements, boxes, container, deadline, max_iters=30
        )
        if refined_vol > best_volume:
            best_volume = refined_vol
            best_placements = refined_placements
    
    if best_placements is None:
        # Fallback
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
