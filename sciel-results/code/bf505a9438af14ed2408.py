"""3D Container Loading - Fast flat-list greedy + multi-phase local search."""
import time
import random
import numpy as np
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


def _fits_at_fast(x, y, z, bw, bh, bd, placed, W, H, D):
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
    candidates = set()
    candidates.add((0, 0, 0))
    if not placed:
        return [(0, 0, 0)]

    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw + bw <= W:
            candidates.add((px + pw, py, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        if px >= bw:
            candidates.add((px - bw, py, pz))
            candidates.add((px - bw, py + ph, pz))
            candidates.add((px - bw, py, pz + pd))
            candidates.add((px - bw, py + ph, pz + pd))
        if pz + pd + bd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
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
        if pz >= bd:
            candidates.add((px, py, pz - bd))
            candidates.add((px + pw, py, pz - bd))
            candidates.add((px, py + ph, pz - bd))
            candidates.add((px + pw, py + ph, pz - bd))

    xs = {0}
    ys = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)

    xs = sorted(x for x in xs if 0 <= x <= W - bw)
    ys = sorted(y for y in ys if 0 <= y <= H - bh)

    if len(xs) <= 15 and len(ys) <= 15:
        for x in xs[:12]:
            for y in ys[:12]:
                min_z = 0
                for (px, py, pz, pw, ph, pd) in placed:
                    if not (x + bw <= px or px + pw <= x):
                        if not (y + bh <= py or py + ph <= y):
                            if pz + pd > min_z:
                                min_z = pz + pd
                if min_z + bd <= D:
                    candidates.add((x, y, min_z))

    valid = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and 0 <= cz and cz + bd <= D:
            valid.append((cx, cy, cz))
    return valid


def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed:
        if x + bw == px:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if x == px + pw:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if y + bh == py:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if y == py + ph:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if z + bd == pz:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
        if z == pz + pd:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    return (z, y, x, -contact)


def _find_best_pos(bw, bh, bd, placed, W, H, D, use_score=True):
    candidates = _generate_candidates(bw, bh, bd, placed, W, H, D)
    best = None
    best_s = None
    for (cx, cy, cz) in candidates:
        if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
            if use_score:
                s = _score_position(cx, cy, cz, bw, bh, bd, placed, W, H, D)
            else:
                s = (cz, cy, cx)
            if best_s is None or s < best_s:
                best_s = s
                best = (cx, cy, cz)
    return best


def _greedy(order, boxes, container, deadline, use_score=True):
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
        best_s = None
        best_rot = None
        for (bw, bh, bd) in rots:
            pos = _find_best_pos(bw, bh, bd, placed, W, H, D, use_score)
            if pos is not None:
                s = (pos[2], pos[1], pos[0])
                if best_s is None or s < best_s:
                    best_s = s
                    best_pos = pos
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


def _vol(result):
    return sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)


def _orders(boxes, n):
    vols = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    maxdims = [max(boxes[i]) for i in range(n)]
    mindims = [min(boxes[i]) for i in range(n)]
    sas = [2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]) for i in range(n)]

    o1 = sorted(range(n), key=lambda i: vols[i], reverse=True)
    o2 = sorted(range(n), key=lambda i: maxdims[i], reverse=True)
    o3 = sorted(range(n), key=lambda i: sas[i], reverse=True)
    o4 = sorted(range(n), key=lambda i: mindims[i], reverse=True)
    o5 = sorted(range(n), key=lambda i: maxdims[i] / max(1, mindims[i]))
    o6 = sorted(range(n), key=lambda i: vols[i] * maxdims[i], reverse=True)
    o7 = sorted(range(n), key=lambda i: sas[i] / max(1, vols[i]))
    o8 = sorted(range(n), key=lambda i: vols[i] * (1 + 0.5 * (1 - maxdims[i] / max(1, mindims[i]))), reverse=True)

    return [o1, o2, o3, o4, o5, o6, o7, o8]


def _build_placed(best, boxes, n, exclude):
    placed = []
    for i in range(n):
        if i not in exclude and best[i][0] == 0:
            x, y, z, bw, bh, bd = best[i][1:7]
            placed.append((x, y, z, bw, bh, bd))
    return placed


def _try_place(idx, boxes, placed, W, H, D, use_score=False):
    w, h, d = boxes[idx]
    rots = _rotations_sorted(w, h, d)
    best_pos = None
    best_s = None
    best_rot = None
    for (bw, bh, bd) in rots:
        pos = _find_best_pos(bw, bh, bd, placed, W, H, D, use_score)
        if pos is not None:
            s = (pos[2], pos[1], pos[0])
            if best_s is None or s < best_s:
                best_s = s
                best_pos = pos
                best_rot = (bw, bh, bd)
    if best_pos is not None:
        return True, best_pos, best_rot
    return False, None, None


def _local_search(result, boxes, container, deadline, max_iters=800):
    W, H, D = container
    n = len(boxes)
    best = [tuple(r) for r in result]
    best_v = _vol(best)
    rng = random.Random(42)

    # Phase 1: Place unplaced boxes
    unplaced = [i for i in range(n) if best[i][0] == -1]
    for _ in range(max(30, len(unplaced) * 4)):
        if time.time() > deadline or not unplaced:
            break
        idx = rng.choice(unplaced)
        placed = _build_placed(best, boxes, n, set())
        ok, pos, rot = _try_place(idx, boxes, placed, W, H, D, False)
        if ok:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            nv = _vol(temp)
            if nv > best_v:
                best = [tuple(r) for r in temp]
                best_v = nv
                unplaced.remove(idx)

    # Phase 2: 1-box relocation
    placed_idx = [i for i in range(n) if best[i][0] == 0]
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if not placed_idx:
            break
        idx = rng.choice(placed_idx)
        placed = _build_placed(best, boxes, n, {idx})
        ok, pos, rot = _try_place(idx, boxes, placed, W, H, D, False)
        if ok:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            nv = _vol(temp)
            if nv > best_v:
                best = [tuple(r) for r in temp]
                best_v = nv

    # Phase 3: 2-box swap
    for _ in range(max_iters // 2):
        if time.time() > deadline:
            break
        if len(placed_idx) < 2:
            break
        i1, i2 = rng.sample(placed_idx, 2)
        placed = _build_placed(best, boxes, n, {i1, i2})
        ok1, pos1, rot1 = _try_place(i1, boxes, placed, W, H, D, False)
        if ok1:
            placed.append((pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2]))
            ok2, pos2, rot2 = _try_place(i2, boxes, placed, W, H, D, False)
            if ok2:
                temp = list(best)
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                nv = _vol(temp)
                if nv > best_v:
                    best = [tuple(r) for r in temp]
                    best_v = nv

    # Phase 4: 3-box swap
    for _ in range(max_iters // 4):
        if time.time() > deadline:
            break
        if len(placed_idx) < 3:
            break
        i1, i2, i3 = rng.sample(placed_idx, 3)
        placed = _build_placed(best, boxes, n, {i1, i2, i3})
        ok1, pos1, rot1 = _try_place(i1, boxes, placed, W, H, D, False)
        if ok1:
            placed.append((pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2]))
            ok2, pos2, rot2 = _try_place(i2, boxes, placed, W, H, D, False)
            if ok2:
                placed.append((pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2]))
                ok3, pos3, rot3 = _try_place(i3, boxes, placed, W, H, D, False)
                if ok3:
                    temp = list(best)
                    temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                    temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                    temp[i3] = (0, pos3[0], pos3[1], pos3[2], rot3[0], rot3[1], rot3[2])
                    nv = _vol(temp)
                    if nv > best_v:
                        best = [tuple(r) for r in temp]
                        best_v = nv

    return best


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115

    if n == 0:
        return []

    orders = _orders(boxes, n)
    best_result = None
    best_vol = -1

    # Greedy with different orderings
    for order in orders:
        if time.time() > deadline - 20:
            break
        res = _greedy(order, boxes, container, deadline, use_score=True)
        v = _vol(res)
        if v > best_vol:
            best_vol = v
            best_result = res

    # Random restarts
    if time.time() < deadline - 10 and best_result is not None:
        rng_r = random.Random(99)
        for _ in range(6):
            if time.time() > deadline - 6:
                break
            ro = list(range(n))
            rng_r.shuffle(ro)
            res = _greedy(ro, boxes, container, deadline, use_score=True)
            v = _vol(res)
            if v > best_vol:
                best_vol = v
                best_result = res

    # Local search
    remaining = deadline - time.time()
    if remaining > 3 and best_result is not None:
        mi = max(200, int(remaining * 60))
        mi = min(mi, 4000)
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=mi)

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
