"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from functools import cmp_to_key

# EVOLVE-BLOCK-START

def _rotations(dims):
    """Return unique rotations of a box (w, h, d)."""
    w, h, d = dims
    seen = set()
    rots = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            rots.append(perm)
    return rots

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with size (w,h,d) fits without overlap and within container."""
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap: intervals overlap on all three axes
        if (x < px + pw and px < x + w and
            y < py + ph and py < y + h and
            z < pz + pd and pz < z + d):
            return False
    return True

def _candidate_positions(placed, W, H, D, w, h, d):
    """Generate candidate (x, y, z) positions using the 'skyline' / extreme points heuristic."""
    # Candidate x, y, z values from existing boxes and 0
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    
    # Filter to those that could fit
    xs = [x for x in xs if x + w <= W]
    ys = [y for y in ys if y + h <= H]
    zs = [z for z in zs if z + d <= D]
    
    if not xs or not ys or not zs:
        return []
    
    # Try all combinations (can be large, but we limit)
    positions = []
    for x in sorted(xs):
        for y in sorted(ys):
            for z in sorted(zs):
                if _fits(placed, x, y, z, w, h, d, W, H, D):
                    positions.append((x, y, z))
    return positions

def _place_box(placed, box_dims, W, H, D, strategy="first_fit", deadline=None):
    """Try to place a box with any rotation. Returns (x,y,z,rotated_dims) or None."""
    rots = _rotations(box_dims)
    
    for rw, rh, rd in rots:
        if strategy == "first_fit":
            positions = _candidate_positions(placed, W, H, D, rw, rh, rd)
            # Sort positions to prefer compact placement (lower x, then y, then z)
            positions.sort(key=lambda p: (p[0], p[1], p[2]))
            for (x, y, z) in positions:
                return (x, y, z, (rw, rh, rd))
        elif strategy == "best_fit":
            # Try to find the position that minimizes "waste" - prefer positions closer to origin
            positions = _candidate_positions(placed, W, H, D, rw, rh, rd)
            if not positions:
                continue
            positions.sort(key=lambda p: (p[0] + p[1] + p[2]))
            for (x, y, z) in positions:
                return (x, y, z, (rw, rh, rd))
    
    return None

def _solve_one(boxes, container, order, strategy, deadline):
    """Solve with a given order and strategy. Returns list of placements."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    box_index_to_placement = [None] * len(boxes)
    
    for idx in order:
        if deadline is not None and time.time() > deadline:
            break
        box = boxes[idx]
        result = _place_box(placed, box, W, H, D, strategy, deadline)
        if result is not None:
            x, y, z, (rw, rh, rd) = result
            placed.append((x, y, z, rw, rh, rd))
            box_index_to_placement[idx] = (x, y, z, rw, rh, rd)
        else:
            box_index_to_placement[idx] = None
    
    # Build output
    placements = []
    for i in range(len(boxes)):
        if box_index_to_placement[i] is not None:
            x, y, z, rw, rh, rd = box_index_to_placement[i]
            placements.append((0, x, y, z, rw, rh, rd))
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    # Compute volume filled
    vol = 0
    for p in placements:
        if p[0] == 0:
            vol += p[4] * p[5] * p[6]
    
    return placements, vol

def _sort_by_volume(boxes):
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

def _sort_by_max_dim(boxes):
    return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)

def _sort_by_min_dim(boxes):
    return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

def _sort_by_base_area(boxes):
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1], reverse=True)

def _random_order(boxes, seed):
    rng = random.Random(seed)
    order = list(range(len(boxes)))
    rng.shuffle(order)
    return order

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_placements = None
    best_vol = 0
    
    # Generate different orderings
    orderings = [
        ("volume_desc", _sort_by_volume(boxes), "first_fit"),
        ("volume_desc", _sort_by_volume(boxes), "best_fit"),
        ("max_dim", _sort_by_max_dim(boxes), "first_fit"),
        ("max_dim", _sort_by_max_dim(boxes), "best_fit"),
        ("min_dim", _sort_by_min_dim(boxes), "first_fit"),
        ("min_dim", _sort_by_min_dim(boxes), "best_fit"),
        ("base_area", _sort_by_base_area(boxes), "first_fit"),
        ("base_area", _sort_by_base_area(boxes), "best_fit"),
    ]
    
    # Add some random orderings
    for seed in range(20):
        if time.time() > deadline:
            break
        order = _random_order(boxes, seed)
        orderings.append((f"random_{seed}", order, "first_fit"))
        orderings.append((f"random_{seed}", order, "best_fit"))
    
    for name, order, strategy in orderings:
        if time.time() > deadline:
            break
        placements, vol = _solve_one(boxes, container, order, strategy, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    if best_placements is None:
        # Fallback: return all as left out
        best_placements = []
        for w, h, d in boxes:
            best_placements.append((-1, 0, 0, 0, w, h, d))
    
    return best_placements

# EVOLVE-BLOCK-END
