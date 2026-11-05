"""3D Container Loading: Optimized with spatial grid and enhanced local search."""
import time
import itertools
import random
import math

def _get_orientations(w, h, d):
    """Return unique orientations of a box (w, h, d)."""
    seen = set()
    orients = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            orients.append(perm)
    return orients

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes strictly overlap."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _generate_candidates(placed, W, H, D, max_candidates=300):
    """
    Generate candidate positions using corner-based approach.
    """
    if not placed:
        return [(0, 0, 0)]
    
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placed:
        # 7 corners of the box (excluding the origin corner)
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
    # Filter valid candidates and sort by priority (low z, then low y, then low x)
    valid_candidates = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
            score = cz * 100000 + cy * 1000 + cx
            valid_candidates.append((score, cx, cy, cz))
            
    valid_candidates.sort(key=lambda x: x[0])
    
    # Limit to max_candidates to avoid explosion
    if len(valid_candidates) > max_candidates:
        valid_candidates = valid_candidates[:max_candidates]
        
    return [c[1:] for c in valid_candidates]

def _pack_greedy(boxes, container, order, deadline, max_candidates=300):
    """
    Fast greedy packing with early termination.
    """
    W, H, D = container
    placed = []
    placements = {}
    filled_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orients = _get_orientations(w, h, d)
        
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        candidates = _generate_candidates(placed, W, H, D, max_candidates)
        
        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    score = cz * 100000 + cy * 1000 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
    return placements, filled_vol

def _try_insert_box(boxes, container, placed, placements, idx, deadline, max_candidates=300):
    """Try to insert a single box. Returns True if successful."""
    W, H, D = container
    w, h, d = boxes[idx]
    orients = _get_orientations(w, h, d)
    
    best_score = float('inf')
    best_pos = None
    best_orient = None
    
    candidates = _generate_candidates(placed, W, H, D, max_candidates)
    
    for ow, oh, od in orients:
        for (cx, cy, cz) in candidates:
            if cx + ow > W or cy + oh > H or cz + od > D:
                continue
            if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                score = cz * 100000 + cy * 1000 + cx
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
    
    if best_pos is not None:
        cx, cy, cz = best_pos
        ow, oh, od = best_orient
        placed.append((cx, cy, cz, ow, oh, od))
        placements[idx] = (cx, cy, cz, ow, oh, od)
        return True
    return False

def _local_search(boxes, container, placements, deadline):
    """
    Local search to improve the placement.
    1. Try to insert any unpacked boxes.
    2. Try to move existing boxes to free up space for larger unpacked boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    # Reconstruct placed list from placements
    placed = []
    for i in placements:
        x, y, z, w, h, d = placements[i]
        placed.append((x, y, z, w, h, d))
        
    filled_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in placements)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to insert any unpacked boxes
        for i in range(n):
            if i in placements:
                continue
            if time.time() > deadline:
                break
            if _try_insert_box(boxes, container, placed, placements, i, deadline):
                filled_vol += boxes[i][0] * boxes[i][1] * boxes[i][2]
                improved = True
                
        # 2. Try to move existing boxes to free up space
        if time.time() > deadline - 5:
            break
            
        for i in list(placements.keys()):
            if time.time() > deadline - 5:
                break
            x, y, z, w, h, d = placements[i]
            box_vol = w * h * d
            
            # Remove box i
            del placements[i]
            placed_without_i = []
            for j in placements:
                px, py, pz, pw, ph, pd = placements[j]
                placed_without_i.append((px, py, pz, pw, ph, pd))
                
            # Try to insert any unpacked box that is larger than the removed one
            inserted = False
            for k in range(n):
                if k in placements:
                    continue
                kw, kh, kd = boxes[k]
                if kw * kh * kd <= box_vol:
                    continue  # Only try to replace with larger or equal volume
                
                if _try_insert_box(boxes, container, placed_without_i, placements, k, deadline):
                    filled_vol += kw * kh * kd - box_vol
                    inserted = True
                    improved = True
                    break
            
            if not inserted:
                # Try to move box i to a better position
                orients = _get_orientations(w, h, d)
                best_score = float('inf')
                best_pos = None
                best_orient = None
                
                candidates = _generate_candidates(placed_without_i, W, H, D, 300)
                
                for ow, oh, od in orients:
                    for (cx, cy, cz) in candidates:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if _fits(placed_without_i, cx, cy, cz, ow, oh, od, W, H, D):
                            score = cz * 100000 + cy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
                
                if best_pos is not None:
                    cx, cy, cz = best_pos
                    ow, oh, od = best_orient
                    placements[i] = (cx, cy, cz, ow, oh, od)
                    placed_without_i.append((cx, cy, cz, ow, oh, od))
                else:
                    # Could not move it, restore original position
                    placements[i] = (x, y, z, w, h, d)
                    placed_without_i.append((x, y, z, w, h, d))
                    
    return placements, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Generate diverse orders
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Surface area descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 5. Min dimension descending
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # 6. Aspect ratio based
    orders.append(sorted(range(n), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 0, reverse=True))
    
    # 7-15. Randomized weighted orders
    random.seed(42)
    for _ in range(9):
        weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        noisy_weights = [w * (1 + random.uniform(-0.2, 0.2)) for w in weights]
        order = list(range(n))
        order.sort(key=lambda i: noisy_weights[i], reverse=True)
        orders.append(order)
        
    best_result = None
    best_filled = 0.0
    
    for i, order in enumerate(orders):
        if time.time() > deadline - 10:  # Keep 10s for local search of last run
            break
            
        # Run greedy
        placements, filled_vol = _pack_greedy(boxes, container, order, deadline - 5)
        
        # Try local search refinement if we have time
        if time.time() < deadline - 5.0:
            placements, filled_vol = _local_search(boxes, container, placements, deadline - 2.0)
            
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements
            
    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
