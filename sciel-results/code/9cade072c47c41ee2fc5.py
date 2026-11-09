"""Combined 3D container loading: extreme-point greedy + multi-heuristic + local search."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes (cartesian product of unique coords)."""
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

def _score_position(pos, box_dims, placed, container):
    """Score a candidate position: prefer compact (low) placements."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Penalize height, then depth, then width (keep things low and compact)
    return (z + d, y + h, x + w)

def _greedy(boxes, container, order, deadline, mode='extreme'):
    """Greedy placement. mode in {'extreme','faces','compact'}."""
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
        
        if mode == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        elif mode == 'faces':
            candidates = _face_points(placed, container)
        else:  # compact
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    if mode == 'compact':
                        key = _score_position(pos, (rw, rh, rd), placed, container)
                    else:
                        # For extreme/faces, prefer lowest z, then y, then x
                        key = (pos[2], pos[1], pos[0])
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

def _local_search(placements, boxes, container, deadline, rng):
    """Try to improve by removing one box and re-placing it elsewhere."""
    W, H, D = container
    n = len(boxes)
    # Build placed list from placements
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    # Try removing and re-inserting random loaded boxes
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    improved = False
    for _ in range(min(30, len(loaded_indices))):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        # Remove box idx
        p = placements[idx]
        box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
        placed.remove(box_pos)
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
                    key = (pos[2], pos[1], pos[0])
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
            placed.append(box_pos)
    
    # Recalculate volume
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Multiple sorting strategies
    sort_keys = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('compactness', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    modes = ['extreme', 'faces', 'compact']
    
    for name, key_fn, reverse in sort_keys:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        for mode in modes:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, mode)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Local search on best
    if best_placements is not None:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng)
    
    # Random restarts with perturbed orders
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for _ in range(min(5, int(remaining_time / 10))):
            if time.time() > deadline:
                break
            # Perturb the order
            perturbed = base_order[:]
            for _ in range(5):
                i, j = rng.sample(range(n), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            mode = rng.choice(modes)
            placements, vol = _greedy(boxes, container, perturbed, deadline, mode)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Final local search if time permits
    if best_placements is not None and time.time() < deadline - 2:
        rng2 = random.Random(999)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2)
    
    return best_placements

# EVOLVE-BLOCK-END
