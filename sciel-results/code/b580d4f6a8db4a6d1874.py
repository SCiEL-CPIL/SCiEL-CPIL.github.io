"""3D Container Loading: multi-start greedy with diverse heuristics and local search."""
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

def _fits(placed, x, y, z, bw, bh, bd, W, H, D):
    """Check if box fits at position without overlap."""
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + bw > px and
            y < py + ph and y + bh > py and
            z < pz + pd and z + bd > pz):
            return False
    return True

def _extreme_points(placed, container):
    """Generate candidate positions from extreme points of placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    xs = sorted(x for x in xs if x < W)
    ys = sorted(y for y in ys if y < H)
    zs = sorted(z for z in zs if z < D)
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_points(placed, container):
    """Generate candidate positions from face-adjacent corners."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        for dx in (0, pw):
            for dy in (0, ph):
                for dz in (0, pd):
                    x, y, z = px + dx, py + dy, pz + dz
                    if x < W and y < H and z < D:
                        candidates.add((x, y, z))
    return list(candidates)

def _score_pos(x, y, z, bw, bh, bd, strategy):
    """Score a position based on strategy."""
    if strategy == 'compact':
        # Prefer positions that minimize the max extent in each dimension
        return (z + bd, y + bh, x + bw)
    elif strategy == 'low':
        # Prefer lower positions
        return (z, y, x)
    elif strategy == 'corner':
        # Prefer corner positions (minimize distance to origin)
        return (x + y + z, z, y, x)
    elif strategy == 'balanced':
        # Prefer positions that balance fill in all dimensions
        return (max(z + bd, y + bh, x + bw), z + bd + y + bh + x + bw)
    else:
        return (z + bd, y + bh, x + bw)

def _place_box(placed, boxes, idx, container, pos_strategy, score_strategy):
    """Try to place box idx at the best position. Returns True if placed."""
    W, H, D = container
    w, h, d = boxes[idx]
    rots = _rotations(w, h, d)
    
    if pos_strategy == 'extreme':
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
    elif pos_strategy == 'faces':
        candidates = _face_points(placed, container)
    elif pos_strategy == 'hybrid':
        # Combine both strategies
        ext = set(_extreme_points(placed, container))
        face = set(_face_points(placed, container))
        candidates = list(ext | face)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
    else:
        candidates = [(0, 0, 0)]
    
    best_key = None
    best_pos = None
    best_rot = None
    
    for pos in candidates:
        x, y, z = pos
        for rw, rh, rd in rots:
            if _fits(placed, x, y, z, rw, rh, rd, W, H, D):
                key = _score_pos(x, y, z, rw, rh, rd, score_strategy)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
                    best_rot = (rw, rh, rd)
    
    if best_pos is not None:
        placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
        return True
    return False

def _greedy_load(boxes, container, order, pos_strategy, score_strategy, deadline):
    """Greedy loading with given order and strategies."""
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = [None] * n
    total_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
        if _place_box(placed, boxes, idx, container, pos_strategy, score_strategy):
            p = placed[-1]
            placements[idx] = (0, p[0], p[1], p[2], p[3], p[4], p[5])
            total_vol += p[3] * p[4] * p[5]
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
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    # Try multiple position/score strategies for re-placement
    strategies = [
        ('extreme', 'compact'),
        ('extreme', 'low'),
        ('faces', 'compact'),
        ('faces', 'low'),
        ('hybrid', 'compact'),
        ('hybrid', 'balanced'),
    ]
    
    for _ in range(min(max_iters, len(loaded_indices) * 3)):
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
        
        # Try to re-place with different strategies
        best_new = None
        best_key = None
        for pos_s, score_s in strategies:
            saved_placed = placed[:]
            if _place_box(placed, boxes, idx, container, pos_s, score_s):
                new_p = placed[-1]
                key = _score_pos(new_p[0], new_p[1], new_p[2], new_p[3], new_p[4], new_p[5], score_s)
                if best_key is None or key < best_key:
                    best_key = key
                    best_new = new_p
                placed = saved_placed
            else:
                placed = saved_placed
        
        if best_new is not None:
            placements[idx] = (0, best_new[0], best_new[1], best_new[2], best_new[3], best_new[4], best_new[5])
        else:
            # Put it back
            placed.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
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
    
    # By surface area descending
    order5 = list(range(n))
    order5.sort(key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(order5)
    
    # Random orders
    rng = random.Random(42)
    for r in range(8):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    best_placements = None
    best_vol = 0.0
    
    strategies = [
        ('extreme', 'compact'),
        ('extreme', 'low'),
        ('faces', 'compact'),
        ('faces', 'low'),
        ('hybrid', 'compact'),
        ('hybrid', 'balanced'),
        ('hybrid', 'corner'),
    ]
    
    # Phase 1: Multi-start greedy
    for order in orders:
        for pos_s, score_s in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy_load(boxes, container, order, pos_s, score_s, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]
    
    # Phase 2: Local search on best
    if best_placements is not None and time.time() < deadline - 5:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_improve(best_placements, boxes, container, deadline, rng2, max_iters=150)
    
    # Phase 3: Additional random restarts if time permits
    if time.time() < deadline - 10:
        rng3 = random.Random(777)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)
        
        num_restarts = 100
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            order = base_order[:]
            num_swaps = rng3.randint(1, min(8, n))
            for _ in range(num_swaps):
                a, b = rng3.randrange(n), rng3.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            
            pos_s = rng3.choice(['extreme', 'faces', 'hybrid'])
            score_s = rng3.choice(['compact', 'low', 'balanced', 'corner'])
            placements, vol = _greedy_load(boxes, container, order, pos_s, score_s, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]
        
        # Final local search
        if best_placements is not None and time.time() < deadline - 2:
            rng4 = random.Random(999)
            best_placements, best_vol = _local_improve(best_placements, boxes, container, deadline, rng4, max_iters=200)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
