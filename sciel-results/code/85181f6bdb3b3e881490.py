"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
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

def _extreme_points(placed, container):
    """Generate candidate positions from extreme points of placed boxes."""
    W, H, D = container
    candidates = {(0, 0, 0)}
    for (px, py, pz, pw, ph, pd) in placed:
        # Points along the 'far' faces of each placed box
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        # Also corners that might create new gaps
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        # Face-based: positions along the far face in one axis, flush in others
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px, py + ph, pz + pd))
    
    # Filter to only valid positions within container
    valid = []
    for (x, y, z) in candidates:
        if x < W and y < H and z < D:
            valid.append((x, y, z))
    return valid

def _score_position(pos, box_dims, placed, container):
    """
    Score a candidate position: prefer positions that leave less wasted space.
    Lower score is better. Uses a combination of:
    - proximity to origin (tight packing)
    - maximization of remaining space in each axis (reduces fragmentation)
    """
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    # Base score: prefer positions closer to origin
    base = x + y + z
    
    # Penalize positions that leave large gaps in any axis
    # Remaining space after placing this box
    rem_x = W - (x + w)
    rem_y = H - (y + h)
    rem_z = D - (z + d)
    
    # We want to minimize the maximum remaining gap (avoids creating large empty regions)
    max_gap = max(rem_x, rem_y, rem_z)
    
    # Combine: prefer tight packing (low base) and minimal max gap
    return base + 0.5 * max_gap

def _score_placement(placed, boxes):
    """Calculate the fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = sum(w * h * d for (_, _, _, w, h, d) in placed)
    return filled / total_vol

def _greedy_place(boxes, container, order, time_limit=0.15):
    """
    Place boxes in the given order using extreme points and greedy selection.
    Returns list of placements (bin, x, y, z, w, h, d) for all boxes.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = [None] * len(boxes)  # final result
    start_time = time.time()
    
    for idx in order:
        if time.time() - start_time > time_limit:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_pos = None
        best_rot = None
        best_score = -1
        
        # Get candidate positions
        candidates = _extreme_points(placed, container)
        
        for (rw, rh, rd) in rots:
            for (cx, cy, cz) in candidates:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                # Score: prefer positions that minimize 'wasted' space
                score = _score_position((cx, cy, cz), (rw, rh, rd), placed, container)
                if best_score < 0 or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            rw, rh, rd = best_rot
            bx, by, bz = best_pos
            placed.append((bx, by, bz, rw, rh, rd))
            placements[idx] = (0, bx, by, bz, rw, rh, rd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes as left out
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

def _random_order(boxes):
    """Generate a random ordering of boxes."""
    order = list(range(len(boxes)))
    random.shuffle(order)
    return order

def _heuristic_order(boxes, container):
    """
    Generate a heuristic ordering: sort by volume descending, 
    with some randomization for ties.
    """
    indexed = list(enumerate(boxes))
    # Sort by volume descending, then by max dimension descending
    indexed.sort(key=lambda x: (-(x[1][0] * x[1][1] * x[1][2]),
                                  -max(x[1])), reverse=False)
    # Actually we want descending volume, so sort by negative volume
    indexed.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    return [idx for idx, _ in indexed]

def _weighted_random_order(boxes, container, alpha=0.5):
    """
    Generate a weighted random order: larger boxes have higher probability
    of being placed earlier, but with some randomness.
    """
    n = len(boxes)
    volumes = [w * h * d for w, h, d in boxes]
    max_vol = max(volumes) if volumes else 1
    if max_vol == 0:
        return _random_order(boxes)
    
    # Create weights: larger boxes get higher weight
    weights = [(v / max_vol) ** alpha for v in volumes]
    
    order = []
    remaining = list(range(n))
    remaining_weights = weights[:]
    
    for _ in range(n):
        # Pick a box with probability proportional to weight
        total_w = sum(remaining_weights)
        if total_w == 0:
            idx = random.choice(remaining)
        else:
            r = random.random() * total_w
            cum = 0
            idx = remaining[-1]
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    idx = remaining[i]
                    break
        order.append(idx)
        remaining.remove(idx)
        remaining_weights.remove(weights[idx])
    
    return order

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    total_box_vol = sum(volumes)
    container_vol = W * H * D
    
    best_placements = None
    best_score = -1
    
    # Strategy 1: Heuristic order (volume descending)
    heuristic_order = _heuristic_order(boxes, container)
    placements = _greedy_place(boxes, container, heuristic_order, time_limit=0.1)
    score = sum(1 for p in placements if p[0] == 0)  # count placed
    # Better: compute actual volume filled
    filled_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    score = filled_vol / container_vol if container_vol > 0 else 0
    if score > best_score:
        best_score = score
        best_placements = placements
    
    # Strategy 2: Multiple weighted random orders
    num_restarts = 60
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        # Vary alpha for diversity
        alpha = 0.3 + 0.4 * (r % 5) / 4.0
        order = _weighted_random_order(boxes, container, alpha=alpha)
        placements = _greedy_place(boxes, container, order, time_limit=0.1)
        filled_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
        score = filled_vol / container_vol if container_vol > 0 else 0
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 3: Pure random orders for more diversity
    for r in range(25):
        if time.time() > deadline:
            break
        order = _random_order(boxes)
        placements = _greedy_place(boxes, container, order, time_limit=0.1)
        filled_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
        score = filled_vol / container_vol if container_vol > 0 else 0
        if score > best_score:
            best_score = score
            best_placements = placements
    
    if best_placements is None:
        # Fallback: no boxes placed
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
# EVOLVE-BLOCK-END
