"""3D Container Loading - Fast EPA greedy + diverse scoring + enhanced local search."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _rotations_sorted(w, h, d):
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
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


def _extreme_points(placed, W, H, D):
    candidates = [(0, 0, 0)]
    if not placed:
        return candidates
    seen = set()
    for (px, py, pz, pw, ph, pd) in placed:
        for cx in (px, px + pw):
            for cy in (py, py + ph):
                for cz in (pz, pz + pd):
                    if cx <= W and cy <= H and cz <= D:
                        if (cx, cy, cz) not in seen:
                            seen.add((cx, cy, cz))
                            candidates.append((cx, cy, cz))
    return candidates


def _score_zyx(pos):
    return (pos[2], pos[1], pos[0])


def _score_sum(pos):
    return (pos[2], pos[0] + pos[1])


def _score_max(pos):
    return (pos[2], max(pos[0], pos[1]))


def _score_contact(pos, bw, bh, bd, placed):
    x, y, z = pos
    contact = 0.0
    for (px, py, pz, pw, ph, pd) in placed:
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
    return (z, y, x, -contact)


def _try_place_box(placed, box_idx, boxes, W, H, D, score_fn=None):
    w, h, d = boxes[box_idx]
    rots = _rotations_sorted(w, h, d)
    candidates = _extreme_points(placed, W, H, D)
    
    best_pos = None
    best_score = None
    best_rot = None
    
    for (bw, bh, bd) in rots:
        for (cx, cy, cz) in candidates:
            if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                if score_fn is not None:
                    s = score_fn((cx, cy, cz), bw, bh, bd, placed)
                else:
                    s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = (cx, cy, cz)
                    best_rot = (bw, bh, bd)
    
    if best_pos is not None:
        return True, best_pos, best_rot
    return False, None, None


def _greedy_place(order, boxes, container, deadline, score_fn=None, rng=None):
    W, H, D = container
    n = len(boxes)
    placed = []
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        success, pos, rot = _try_place_box(placed, idx, boxes, W, H, D, score_fn)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            placed.append((x, y, z, bw, bh, bd))
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            w, h, d = boxes[idx]
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _compute_volume(result):
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _heuristic_orders(boxes, n):
    orders = []
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(('vol', vol_order))
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]))
    orders.append(('mindim', mindim_order))
    
    sumdim_order = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    orders.append(('sumdim', sumdim_order))
    
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))
    
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    
    hybrid_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * (1 + 0.5 * (1 - max(boxes[i])/max(1, min(boxes[i])))), reverse=True)
    orders.append(('hybrid', hybrid_order))
    
    return orders


def _local_search(result, boxes, container, deadline, max_iters=500):
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    def get_placed_list(res):
        placed = []
        for i in range(n):
            if res[i][0] == 0:
                x, y, z, bw, bh, bd = res[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed
    
    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 4 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed_list(best_result)
        success, pos, rot = _try_place_box(placed, idx, boxes, W, H, D)
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
    
    # Phase 2: Single-box relocation
    for _ in range(max_iters):
        if time.time() > deadline:
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
        success, pos, rot = _try_place_box(placed, idx, boxes, W, H, D)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
    
    # Phase 3: 2-box swap
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    for _ in range(max_iters // 2):
        if time.time() > deadline:
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
        success1, pos1, rot1 = _try_place_box(placed, i1, boxes, W, H, D)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = _try_place_box(placed, i2, boxes, W, H, D)
            if success2:
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
                    best_vol = new_vol
    
    # Phase 4: 3-box swap
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    for _ in range(max_iters // 3):
        if time.time() > deadline:
            break
        if len(placed_indices) < 3:
            break
        i1, i2, i3 = rng.sample(placed_indices, 3)
        temp = list(best_result)
        for ii in (i1, i2, i3):
            temp[ii] = (-1, 0, 0, 0, boxes[ii][0], boxes[ii][1], boxes[ii][2])
        placed = []
        for i in range(n):
            if i not in (i1, i2, i3) and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        success1, pos1, rot1 = _try_place_box(placed, i1, boxes, W, H, D)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = _try_place_box(placed, i2, boxes, W, H, D)
            if success2:
                placed.append(pos2 + rot2)
                success3, pos3, rot3 = _try_place_box(placed, i3, boxes, W, H, D)
                if success3:
                    temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                    temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                    temp[i3] = (0, pos3[0], pos3[1], pos3[2], rot3[0], rot3[1], rot3[2])
                    new_vol = _compute_volume(temp)
                    if new_vol > best_vol:
                        best_result = [tuple(r) for r in temp]
                        best_vol = new_vol
    
    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run greedy with different orderings and scoring functions
    score_fns = [None, _score_sum, _score_max, _score_contact]
    
    for name, order in heuristic_orders:
        if time.time() > deadline - 25:
            break
        for sf in score_fns:
            if time.time() > deadline - 25:
                break
            result = _greedy_place(order, boxes, container, deadline, sf)
            vol = _compute_volume(result)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Random restarts
    if time.time() < deadline - 12 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(12):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            sf = score_fns[rng_restart.randint(0, len(score_fns)-1)]
            result_rand = _greedy_place(random_order, boxes, container, deadline, sf)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 60)
        max_iters = max(150, min(max_iters, 3000))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
