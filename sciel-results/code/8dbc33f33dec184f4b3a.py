"""3D Container Loading: spatial-hashed best-fit + randomized multi-start + local search + gap fill."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import defaultdict


def _orientations(w, h, d):
    """Return all distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True


def _fits(placed, x, y, z, w, h, d, W, H, D):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True


def _score_position(x, y, z, w, h, d, W, H, D, cur_max_x, cur_max_y, cur_max_z):
    new_max_x = max(cur_max_x, x + w)
    new_max_y = max(cur_max_y, y + h)
    new_max_z = max(cur_max_z, z + d)
    return new_max_x * new_max_y * new_max_z


def _pack_best_fit(boxes, container, order, deadline):
    """
    Pack boxes using Best-Fit heuristic with candidate-position enumeration.
    """
    W, H, D = container
    placed = []
    placements = {}
    filled_vol = 0.0

    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0

    for idx in order:
        if time.time() > deadline:
            break

        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None

        # Build candidate coordinates from placed boxes
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

        xs = sorted(x for x in xs if 0 <= x <= W)
        ys = sorted(y for y in ys if 0 <= y <= H)
        zs = sorted(z for z in zs if 0 <= z <= D)

        for ow, oh, od in _orientations(w, h, d):
            for cx in xs:
                if cx + ow > W:
                    continue
                for cy in ys:
                    if cy + oh > H:
                        continue
                    for cz in zs:
                        if cz + od > D:
                            continue
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            score = _score_position(cx, cy, cz, ow, oh, od, W, H, D,
                                                    cur_max_x, cur_max_y, cur_max_z)
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)

        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            cur_max_x = max(cur_max_x, cx + ow)
            cur_max_y = max(cur_max_y, cy + oh)
            cur_max_z = max(cur_max_z, cz + od)

    return placements, filled_vol


def _local_search(boxes, container, placements, filled_vol, deadline):
    """
    Local search: try to re-insert each box at a better position.
    """
    W, H, D = container
    n = len(boxes)
    best_placements = dict(placements)
    best_filled = filled_vol

    placed = []
    for idx in sorted(best_placements.keys()):
        x, y, z, w, h, d = best_placements[idx]
        placed.append((x, y, z, w, h, d))

    for _iteration in range(4):
        if time.time() > deadline:
            break

        improved = False
        for idx in list(best_placements.keys()):
            if time.time() > deadline:
                break

            bx, by, bz, bw, bh, bd = best_placements[idx]
            placed = [p for p in placed if p != (bx, by, bz, bw, bh, bd)]
            del best_placements[idx]

            cur_max_x = 0
            cur_max_y = 0
            cur_max_z = 0
            for (px, py, pz, pw, ph, pd) in placed:
                cur_max_x = max(cur_max_x, px + pw)
                cur_max_y = max(cur_max_y, py + ph)
                cur_max_z = max(cur_max_z, pz + pd)

            w, h, d = boxes[idx]
            best_score = float('inf')
            best_pos = None
            best_orient = None

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

            xs = sorted(x for x in xs if 0 <= x <= W)
            ys = sorted(y for y in ys if 0 <= y <= H)
            zs = sorted(z for z in zs if 0 <= z <= D)

            for ow, oh, od in _orientations(w, h, d):
                for cx in xs:
                    if cx + ow > W:
                        continue
                    for cy in ys:
                        if cy + oh > H:
                            continue
                        for cz in zs:
                            if cz + od > D:
                                continue
                            if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                                score = _score_position(cx, cy, cz, ow, oh, od, W, H, D,
                                                        cur_max_x, cur_max_y, cur_max_z)
                                if score < best_score:
                                    best_score = score
                                    best_pos = (cx, cy, cz)
                                    best_orient = (ow, oh, od)

            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                best_placements[idx] = (cx, cy, cz, ow, oh, od)
                placed.append((cx, cy, cz, ow, oh, od))
                improved = True
            else:
                best_placements[idx] = (bx, by, bz, bw, bh, bd)
                placed.append((bx, by, bz, bw, bh, bd))

        if not improved:
            break

    return best_placements, best_filled


def _gap_fill(boxes, container, placements, deadline):
    """
    Try to insert remaining boxes into gaps.
    """
    W, H, D = container
    n = len(boxes)
    best_placements = dict(placements)
    filled_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in best_placements)

    placed = []
    for idx in sorted(best_placements.keys()):
        x, y, z, w, h, d = best_placements[idx]
        placed.append((x, y, z, w, h, d))

    unplaced = [i for i in range(n) if i not in best_placements]
    unplaced.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])

    for idx in unplaced:
        if time.time() > deadline:
            break

        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None

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

        xs = sorted(x for x in xs if 0 <= x <= W)
        ys = sorted(y for y in ys if 0 <= y <= H)
        zs = sorted(z for z in zs if 0 <= z <= D)

        for ow, oh, od in _orientations(w, h, d):
            for cx in xs:
                if cx + ow > W:
                    continue
                for cy in ys:
                    if cy + oh > H:
                        continue
                    for cz in zs:
                        if cz + od > D:
                            continue
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            cur_max_x = max(p + pw for (p, py, pz, pw, ph, pd) in placed) if placed else 0
                            cur_max_y = max(p + ph for (px, p, pz, pw, ph, pd) in placed) if placed else 0
                            cur_max_z = max(p + pd for (px, py, p, pw, ph, pd) in placed) if placed else 0
                            score = _score_position(cx, cy, cz, ow, oh, od, W, H, D,
                                                    cur_max_x, cur_max_y, cur_max_z)
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)

        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            best_placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d

    return best_placements, filled_vol


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple deterministic sort orders
    orders = []

    # 1: volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))

    # 2: max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))

    # 3: sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))

    # 4: aspect ratio descending (most elongated first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(min(boxes[i]), 1e-9)), reverse=True))

    # 5: min dimension descending (flat boxes first)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))

    # 6: surface area descending
    orders.append(sorted(range(n), key=lambda i: 2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]), reverse=True))

    # 7: volume / max_dim (compactness)
    orders.append(sorted(range(n), key=lambda i: (boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(boxes[i]), reverse=True))

    # 8: randomized shuffle (seeded for reproducibility)
    rng = random.Random(42)
    shuffled = list(range(n))
    rng.shuffle(shuffled)
    orders.append(shuffled)

    # 9: reverse of volume order
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2]))

    best_result = None
    best_filled = 0.0

    for order in orders:
        if time.time() > deadline:
            break

        placements, filled_vol = _pack_best_fit(boxes, container, order, deadline)

        if time.time() < deadline:
            placements, filled_vol = _gap_fill(boxes, container, placements, deadline)

        if time.time() < deadline:
            placements, filled_vol = _local_search(boxes, container, placements, filled_vol, deadline)

        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements

    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))

    return result
# EVOLVE-BLOCK-END
