"""3D Container Loading: Multi-strategy greedy with local search improvement."""
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_check(placed_list, x, y, z, bw, bh, bd, W, H, D):
    """Check if box fits at position without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed_list:
        if (x < px + pw and x + bw > px and
            y < py + ph and y + bh > py and
            z < pz + pd and z + bd > pz):
            return False
    return True

def _extreme_positions(placed_list, W, H, D):
    """Generate candidate positions from all combinations of face coordinates."""
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed_list:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    xs = sorted(x for x in xs if 0 <= x < W)
    ys = sorted(y for y in ys if 0 <= y < H)
    zs = sorted(z for z in zs if 0 <= z < D)
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_positions(placed_list, W, H, D):
    """Generate candidate positions from face-adjacent corners of placed boxes."""
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed_list:
        for dx in (0, pw):
            for dy in (0, ph):
                for dz in (0, pd):
                    x, y, z = px + dx, py + dy, pz + dz
                    if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                        candidates.add((x, y, z))
    return list(candidates)

def _place_box(placed_list, box_dims, container, pos_strategy, score_strategy, max_candidates=300):
    """Try to place a box at the best position. Returns (x,y,z,rw,rh,rd) or None."""
    W, H, D = container
    w, h, d = box_dims
    rots = _rotations(w, h, d)
    
    if pos_strategy == 'extreme':
        candidates = _extreme_positions(placed_list, W, H, D)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
    else:
        candidates = _face_positions(placed_list, W, H, D)
    
    # Limit candidates for performance
    if len(candidates) > max_candidates:
        if score_strategy == 'compact':
            # Sort by compactness (minimize max extent)
            candidates.sort(key=lambda p: max(p[0], p[1], p[2]))
        else:
            candidates.sort(key=lambda p: p[0] + p[1] + p[2])
        candidates = candidates[:max_candidates]
    
    best_key = None
    best_pos = None
    best_rot = None
    
    for pos in candidates:
        x, y, z = pos
        for rw, rh, rd in rots:
            if _fits_check(placed_list, x, y, z, rw, rh, rd, W, H, D):
                if score_strategy == 'compact':
                    # Prefer positions that minimize the bounding box of all placed boxes
                    key = (z + rd, y + rh, x + rw)
                else:
                    key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
                    best_rot = (rw, rh, rd)
    
    if best_pos is not None:
        return best_pos + best_rot
    return None

def _greedy_load(boxes, container, order, pos_strategy, score_strategy, deadline):
    """Greedy loading with given order and strategies."""
    W, H, D = container
    n = len(boxes)
    placed_list = []
    placements = [None] * n
    total_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
        result = _place_box(placed_list, boxes[idx], container, pos_strategy, score_strategy)
        if result is not None:
            x, y, z, rw, rh, rd = result
            placed_list.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            w, h, d = boxes[idx]
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_improve(placements, boxes, container, deadline, rng, max_iters=100):
    """Local search: remove and re-insert boxes to improve packing."""
    W, H, D = container
    n = len(boxes)
    
    placed_list = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed_list.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    best_placements = [p[:] for p in placements]
    best_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if not loaded_indices:
            break
        
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove
        try:
            placed_list.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        # Try to re-place with different strategies
        placed = False
        for pos_s, score_s in [('extreme', 'compact'), ('faces', 'compact'), ('extreme', 'low'), ('faces', 'low')]:
            result = _place_box(placed_list, boxes[idx], container, pos_s, score_s)
            if result is not None:
                x, y, z, rw, rh, rd = result
                placed_list.append((x, y, z, rw, rh, rd))
                placements[idx] = (0, x, y, z, rw, rh, rd)
                placed = True
                break
        
        if not placed:
            # Put it back
            placed_list.append(box_entry)
        else:
            # Check if we improved
            cur_vol = sum(pl[i][4] * pl[i][5] * pl[i][6] for i in range(n) if pl[i][0] == 0)
            if cur_vol > best_vol:
                best_vol = cur_vol
                best_placements = [p[:] for p in placements]
            else:
                # Revert
                placed_list.remove((x, y, z, rw, rh, rd))
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    return best_placements, best_vol

def _get_orders(boxes, container, rng):
    """Generate diverse ordering strategies."""
    W, H, D = container
    n = len(boxes)
    orders = []
    
    vols = [w * h * d for w, h, d in boxes]
    
    # By volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: vols[i], reverse=True)
    orders.append(order1)
    
    # By max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: max(boxes[i]), reverse=True)
    orders.append(order2)
    
    # By min dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: min(boxes[i]), reverse=True)
    orders.append(order3)
    
    # By aspect ratio (most cubic first)
    order4 = list(range(n))
    order4.sort(key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(order4)
    
    # By sum of dimensions descending
    order5 = list(range(n))
    order5.sort(key=lambda i: sum(boxes[i]), reverse=True)
    orders.append(order5)
    
    # Random orders
    for _ in range(5):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    return orders

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    orders = _get_orders(boxes, container, rng)
    
    strategies = [
        ('extreme', 'compact'),
        ('extreme', 'low'),
        ('faces', 'compact'),
        ('faces', 'low'),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    # Phase 1: Try all order/strategy combinations
    for order in orders:
        for pos_s, score_s in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy_load(boxes, container, order, pos_s, score_s, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        # Fallback: return all boxes as not loaded
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        improved_placements, improved_vol = _local_improve(
            best_placements, boxes, container, deadline, rng2, max_iters=150
        )
        if improved_vol > best_vol:
            best_vol = improved_vol
            best_placements = improved_placements
    
    # Phase 3: Additional random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        rng3 = random.Random(789)
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            ro = list(range(n))
            rng3.shuffle(ro)
            for pos_s, score_s in strategies[:2]:  # Only best 2 strategies
                if time.time() > deadline - 5:
                    break
                placements, vol = _greedy_load(boxes, container, ro, pos_s, score_s, deadline)
                if vol > best_vol:
                    best_vol = vol
                    best_placements = [p[:] for p in placements]
                    # Quick local search on new best
                    if time.time() < deadline - 10:
                        rng4 = random.Random(456)
                        imp_pl, imp_vol = _local_improve(
                            best_placements, boxes, container, deadline, rng4, max_iters=50
                        )
                        if imp_vol > best_vol:
                            best_vol = imp_vol
                            best_placements = imp_pl
    
    return best_placements
