"""3D Container Loading: combined extreme-point greedy + multi-heuristic + local search."""
import time
import random
import itertools
import sys

def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_points(placed, container):
    """Generate candidate positions from face-adjacent corners of placed boxes."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    return [(x, y, z) for (x, y, z) in candidates if x < W and y < H and z < D]

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _score_position_compact(pos, box_dims, placed, container):
    """Score a candidate position: prefer compact (low) placements."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Penalize height, then depth, then width (keep things low and compact)
    return (z + d, y + h, x + w)

def _score_position_low(pos, box_dims, placed, container):
    """Score a candidate position: prefer lowest z, then y, then x."""
    return (pos[2], pos[1], pos[0])

def _greedy(boxes, container, order, deadline, pos_strategy, score_strategy):
    """Greedy placement with given position and scoring strategies."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if pos_strategy == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        else:  # faces
            candidates = _face_points(placed, container)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    if score_strategy == 'compact':
                        key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    else:
                        key = _score_position_low(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
    """Local search: remove and re-insert boxes to improve packing."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    improved = False
    for _ in range(min(max_iters, len(loaded_indices) * 2)):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove
        try:
            placed.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Try to place it back at a better position
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            improved = True
        else:
            # Put it back
            placed.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Generate diverse orders
    orders = []
    
    # By volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: vols[i], reverse=True)
    orders.append(order1)
    
    # By min dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: min(boxes[i]), reverse=True)
    orders.append(order2)
    
    # By max dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: max(boxes[i]), reverse=True)
    orders.append(order3)
    
    # By aspect ratio (flat boxes first)
    order4 = list(range(n))
    order4.sort(key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(order4)
    
    # Random orders
    rng = random.Random(42)
    for r in range(5):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    best_placements = None
    best_vol = 0.0
    
    strategies = [('extreme', 'compact'), ('extreme', 'low'), ('faces', 'compact'), ('faces', 'low')]
    
    for order in orders:
        for pos_s, score_s in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2)
    
    # Additional random restarts if time permits
    if time.time() < deadline - 20:
        rng3 = random.Random(777)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            order = base_order[:]
            num_swaps = rng3.randint(1, min(5, n))
            for _ in range(num_swaps):
                a, b = rng3.randrange(n), rng3.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            
            pos_s = rng3.choice(['extreme', 'faces'])
            score_s = rng3.choice(['compact', 'low'])
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
        
        # Final local search
        if best_placements is not None and time.time() < deadline - 2:
            rng4 = random.Random(999)
            best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng4)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
