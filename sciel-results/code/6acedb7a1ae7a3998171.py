import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, W, H, D):
    """Generate candidate (x,y,z) positions from placed boxes using cross-product of face coordinates."""
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)
    
    # Filter to those within container bounds
    xs = [x for x in xs if 0 <= x < W]
    ys = [y for y in ys if 0 <= y < H]
    zs = [z for z in zs if 0 <= z < D]
    
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _fits(box_dims, pos, W, H, D, placed):
    """Check if box fits at pos without overlap and within container."""
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

def _greedy(boxes, container, order, deadline):
    """Greedy placement using extreme points for a given order. Returns (placements, volume)."""
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
        
        # Generate candidate positions
        candidates = _extreme_points(placed, W, H, D)
        if not candidates:
            candidates = [(0, 0, 0)]
        
        # Sort candidates by proximity to origin (compact packing)
        candidates.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
        
        best_pos = None
        best_rot = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, W, H, D, placed):
                    best_pos = pos
                    best_rot = (rw, rh, rd)
                    break
            if best_pos is not None:
                break
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    W, H, D = container
    n = len(boxes)
    
    # Precompute volumes and sort indices
    vols = [w * h * d for w, h, d in boxes]
    
    # Initial greedy: largest first
    order = list(range(n))
    order.sort(key=lambda i: vols[i], reverse=True)
    
    best_placements, best_vol = _greedy(boxes, container, order, deadline)
    
    # Local search: try multiple random orderings
    rng = random.Random(42)
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        # Create a perturbed order: mostly sorted, with some swaps
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        # Randomly swap a few elements
        for _ in range(min(10, n)):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        placements, vol = _greedy(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    return best_placements
