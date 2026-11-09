"""3D Container Loading - Skyline + Shelf hybrid with fast candidates."""
import time
import random
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _rotations_sorted(w, h, d):
    """Return unique rotations sorted by max-dim ascending (more compact first)."""
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


def _fits_at_fast(x, y, z, bw, bh, bd, placed, W, H, D):
    """Faster fit check using early termination."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for i in range(len(placed)):
        px, py, pz, pw, ph, pd = placed[i]
        if px + pw <= x:
            continue
        if x + bw <= px:
            continue
        if py + ph <= y:
            continue
        if y + bh <= py:
            continue
        if pz + pd <= z:
            continue
        if z + bd <= pz:
            continue
        return False
    return True


def _generate_candidates(bw, bh, bd, placed, W, H, D):
    """Generate candidate (x, y, z) positions for a box of given size."""
    candidates = set()
    candidates.add((0, 0, 0))
    
    if not placed:
        return [(0, 0, 0)]
    
    # Generate positions based on existing box boundaries (extreme points)
    for (px, py, pz, pw, ph, pd) in placed:
        # Right of box
        if px + pw + bw <= W:
            candidates.add((px + pw, py, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # Left of box (if box not at x=0)
        if px >= bw:
            candidates.add((px - bw, py, pz))
            candidates.add((px - bw, py + ph, pz))
            candidates.add((px - bw, py, pz + pd))
            candidates.add((px - bw, py + ph, pz + pd))
        # Above box
        if pz + pd + bd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # In front/behind in y
        if py + ph + bh <= H:
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        if py >= bh:
            candidates.add((px, py - bh, pz))
            candidates.add((px + pw, py - bh, pz))
            candidates.add((px, py - bh, pz + pd))
            candidates.add((px + pw, py - bh, pz + pd))
        # Behind in z
        if pz + pd + bd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        if pz >= bd:
            candidates.add((px, py, pz - bd))
            candidates.add((px + pw, py, pz - bd))
            candidates.add((px, py + ph, pz - bd))
            candidates.add((px + pw, py + ph, pz - bd))
    
    # Grid-based positions for diversity (skyline-like)
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
    
    xs = sorted(set(x for x in xs if 0 <= x <= W - bw))
    ys = sorted(set(y for y in ys if 0 <= y <= H - bh))
    zs = sorted(set(z for z in zs if 0 <= z <= D - bd))
    
    # Limit grid search to keep it fast
    if len(xs) <= 15 and len(ys) <= 15:
        for x in xs[:10]:
            for y in ys[:10]:
                min_z = 0
                for (px, py, pz, pw, ph, pd) in placed:
                    if not (x + bw <= px or px + pw <= x):
                        if not (y + bh <= py or py + ph <= y):
                            if pz + pd > min_z:
                                min_z = pz + pd
                if min_z + bd <= D:
                    candidates.add((x, y, min_z))
    
    # Filter candidates
    valid = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and 0 <= cz and cz + bd <= D:
            valid.append((cx, cy, cz))
    
    return valid


def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position using smooth weighted surrogate: lower is better."""
    # Weighted sum: z dominates, then y, then x, then -contact
    # Using smooth weights to create a more continuous landscape
    contact = 0.0
    for (px, py, pz, pw, ph, pd) in placed:
        # Check face contacts
        if abs((x + bw) - px) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs((y + bh) - py) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs((z + bd) - pz) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    
    # Smooth weighted score: z has highest weight, then y, x, then -contact
    # This creates a smoother optimization landscape than lexicographic
    score = z * 1.0 + y * 0.01 + x * 0.001 - contact * 0.0001
    return score


def _fast_greedy_place(order, boxes, container, deadline):
    """Fast greedy placement using candidate generation."""
    W, H, D = container
    n = len(boxes)
    placed = []
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            candidates = _generate_candidates(bw, bh, bd, placed, W, H, D)
            for (cx, cy, cz) in candidates:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _heuristic_orders(boxes, n):
    """Generate multiple heuristic orderings."""
    orders = []
    
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(('vol', vol_order))
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(('mindim', mindim_order))
    
    # Compactness: prefer more cube-like boxes first
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))
    
    # Volume-weighted by max dimension
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    
    # Surface area to volume ratio (prefer lower ratio = more compact)
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]))
    orders.append(('svr', svr_order))
    
    # Hybrid: volume * (1 + compactness bonus)
    hybrid_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * (1 + 0.5 * (1 - max(boxes[i])/max(1, min(boxes[i])))), reverse=True)
    orders.append(('hybrid', hybrid_order))
    
    # Shelf-like: sort by height (d) descending to create shelf layers
    shelf_height_order = sorted(range(n), key=lambda i: boxes[i][2], reverse=True)
    orders.append(('shelf_h', shelf_height_order))
    
    # Shelf-like: sort by volume/height ratio (wider shelves first)
    shelf_ratio_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1] / max(1, boxes[i][2]), reverse=True)
    orders.append(('shelf_ratio', shelf_ratio_order))
    
    # Shelf-like: sort by max(w,h) descending (wide boxes first for shelf base)
    shelf_wide_order = sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1]), reverse=True)
    orders.append(('shelf_wide', shelf_wide_order))
    
    return orders


def _try_place_box(idx, placed, boxes, W, H, D):
    """Try to place box idx given current placed boxes."""
    w, h, d = boxes[idx]
    rots = _rotations_sorted(w, h, d)
    
    best_pos = None
    best_score = None
    best_rot = None
    
    for (bw, bh, bd) in rots:
        candidates = _generate_candidates(bw, bh, bd, placed, W, H, D)
        for (cx, cy, cz) in candidates:
            if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = (cx, cy, cz)
                    best_rot = (bw, bh, bd)
    
    if best_pos is not None:
        return True, best_pos, best_rot
    return False, None, None


def _local_search(result, boxes, container, deadline, max_iters=500):
    """Local search: try to place unplaced boxes and improve positions of placed boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    def get_placed_list(result):
        placed = []
        for i in range(n):
            if result[i][0] == 0:
                x, y, z, bw, bh, bd = result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed
    
    # Phase 1: Try to place unplaced boxes (multiple passes)
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    max_attempts = max(max_iters // 2, 200)
    while unplaced and attempts < max_attempts and time.time() < deadline - 5:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed_list(best_result)
        success, pos, rot = _try_place_box(idx, placed, boxes, W, H, D)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best_result)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
                unplaced.remove(idx)
    
    # Phase 2: Move placed boxes to find better positions
    # Use simulated annealing-like acceptance for better exploration
    temperature = max(1.0, best_vol * 0.01)
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        temp = list(best_result)
        temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        placed = []
        for i in range(n):
            if i != idx and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        success, pos, rot = _try_place_box(idx, placed, boxes, W, H, D)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            delta_vol = new_vol - best_vol
            # Accept if better, or with probability based on temperature
            if delta_vol > 0 or (temperature > 0.1 and rng.random() < np.exp(delta_vol / temperature)):
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
            # Cool down
            temperature *= 0.995
    
    # Phase 3: 2-box swap moves
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    swap_iters = max(max_iters // 3, 100)
    for _ in range(swap_iters):
        if time.time() > deadline - 1:
            break
        if len(placed_indices) < 2:
            break
        
        i1, i2 = rng.sample(placed_indices, 2)
        temp = list(best_result)
        temp[i1] = (-1, 0, 0, 0, boxes[i1][0], boxes[i1][1], boxes[i1][2])
        temp[i2] = (-1, 0, 0, 0, boxes[i2][0], boxes[i2][1], boxes[i2][2])
        
        placed = []
        for i in range(n):
            if i not in (i1, i2) and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        success1, pos1, rot1 = _try_place_box(i1, placed, boxes, W, H, D)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = _try_place_box(i2, placed, boxes, W, H, D)
            if success2:
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
                    best_vol = new_vol
    
    # Phase 4: Final polish - try to re-place all boxes with best scoring
    # This is a high-precision polish pass
    if time.time() < deadline - 2:
        final_result = [tuple(r) for r in best_result]
        final_vol = _compute_volume(final_result)
        # Try to improve each placed box position
        for idx in range(n):
            if time.time() > deadline - 1:
                break
            if final_result[idx][0] == 0:
                temp = list(final_result)
                temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                placed = []
                for i in range(n):
                    if i != idx and temp[i][0] == 0:
                        x, y, z, bw, bh, bd = temp[i][1:7]
                        placed.append((x, y, z, bw, bh, bd))
                success, pos, rot = _try_place_box(idx, placed, boxes, W, H, D)
                if success:
                    x, y, z = pos
                    bw, bh, bd = rot
                    temp[idx] = (0, x, y, z, bw, bh, bd)
                    new_vol = _compute_volume(temp)
                    if new_vol > final_vol:
                        final_result = [tuple(r) for r in temp]
                        final_vol = new_vol
        if final_vol > best_vol:
            best_result = final_result
            best_vol = final_vol
    
    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115  # Safety margin
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run fast greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline - 30:
            break
        result = _fast_greedy_place(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Random restarts
    if time.time() < deadline - 15 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _fast_greedy_place(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 50)
        max_iters = max(200, min(max_iters, 3000))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
