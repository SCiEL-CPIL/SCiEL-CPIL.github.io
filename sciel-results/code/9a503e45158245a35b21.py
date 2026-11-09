import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    W, H, D = container
    n = len(boxes)

    def get_orientations(w, h, d):
        return list(set(itertools.permutations([w, h, d])))

    # ---- Spatial hash for fast collision queries ----
    def make_grid(placed):
        g = defaultdict(list)
        for (idx, x, y, z, w, h, d) in placed:
            x0, y0, z0 = int(x), int(y), int(z)
            x1, y1, z1 = int(x + w), int(y + h), int(z + d)
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    for gz in range(z0, z1 + 1):
                        g[(gx, gy, gz)].append((idx, x, y, z, w, h, d))
        return g

    def query_candidates(g, cx, cy, cz, ow, oh, od):
        # Any placed box whose cell range intersects the candidate AABB.
        x0, y0, z0 = int(cx), int(cy), int(cz)
        x1, y1, z1 = int(cx + ow), int(cy + oh), int(cz + od)
        seen = set()
        out = []
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                for gz in range(z0, z1 + 1):
                    for item in g.get((gx, gy, gz), ()):
                        if item[0] not in seen:
                            seen.add(item[0])
                            out.append(item)
        return out

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    # ---- Skyline corner generator (stronger than basic corners) ----
    def corner_candidates(placed, ow, oh, od):
        xs = {0.0}
        ys = {0.0}
        zs = {0.0}
        for (_, x, y, z, w, h, d) in placed:
            xs.add(x); xs.add(x + w)
            ys.add(y); ys.add(y + h)
            zs.add(z); zs.add(z + d)
        cands = []
        for x in xs:
            if x + ow > W:
                continue
            for y in ys:
                if y + oh > H:
                    continue
                for z in zs:
                    if z + od > D:
                        continue
                    cands.append((x, y, z))
        return cands

    def greedy_pack(order):
        # order: list of box indices in placement order
        placed = []
        grid = {}
        total_vol = 0
        for idx in order:
            w, h, d = boxes[idx]
            best = None
            best_score = None
            for ow, oh, od in get_orientations(w, h, d):
                for (cx, cy, cz) in corner_candidates(placed, ow, oh, od):
                    # quick bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    # collision
                    ok = True
                    for item in query_candidates(grid, cx, cy, cz, ow, oh, od):
                        if overlaps(cx, cy, cz, ow, oh, od, item[1], item[2], item[3], item[4], item[5], item[6]):
                            ok = False
                            break
                    if not ok:
                        continue
                    # score: prefer low, tight packing; z (height) most costly
                    score = cx + cy + 2.0 * cz
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (idx, cx, cy, cz, ow, oh, od)
            if best is not None:
                placed.append(best)
                total_vol += best[4] * best[5] * best[6]
                grid = make_grid(placed)
        return placed, total_vol

    def reinsert(placed, total_vol):
        # Try to place the boxes that were left out, in a couple of orders.
        placed_idx = set(p[0] for p in placed)
        left = [i for i in range(n) if i not in placed_idx]
        grid = make_grid(placed)
        improved = True
        while improved:
            improved = False
            for left in (sorted(left, key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2]),
                         sorted(left, key=lambda i: max(boxes[i]))):
                new_placed = list(placed)
                new_vol = total_vol
                for idx in left:
                    if idx in set(p[0] for p in new_placed):
                        continue
                    w, h, d = boxes[idx]
                    best = None
                    best_score = None
                    for ow, oh, od in get_orientations(w, h, d):
                        for (cx, cy, cz) in corner_candidates(new_placed, ow, oh, od):
                            if cx + ow > W or cy + oh > H or cz + od > D:
                                continue
                            ok = True
                            for item in query_candidates(grid, cx, cy, cz, ow, oh, od):
                                if overlaps(cx, cy, cz, ow, oh, od, item[1], item[2], item[3], item[4], item[5], item[6]):
                                    ok = False
                                    break
                            if not ok:
                                continue
                            score = cx + cy + 2.0 * cz
                            if best_score is None or score < best_score:
                                best_score = score
                                best = (idx, cx, cy, cz, ow, oh, od)
                    if best is not None:
                        new_placed.append(best)
                        new_vol += best[4] * best[5] * best[6]
                if new_vol > total_vol:
                    placed = new_placed
                    total_vol = new_vol
                    grid = make_grid(placed)
                    improved = True
        return placed, total_vol

    def to_result(placed):
        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for (idx, x, y, z, w, h, d) in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, sum(p[4] * p[5] * p[6] for p in placed)

    # ---- Build candidate orders ----
    base_orders = []
    base_orders.append(sorted(range(n), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2]))
    base_orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
    base_orders.append(sorted(range(n), key=lambda i: -(boxes[i][0] + boxes[i][1] + boxes[i][2])))
    base_orders.append(sorted(range(n), key=lambda i: -(max(boxes[i]) / max(min(boxes[i]), 1e-9))))

    # random orders
    rng = random.Random(12345)
    for _ in range(8):
        o = list(range(n))
        rng.shuffle(o)
        base_orders.append(o)

    best_result = None
    best_vol = 0

    for order in base_orders:
        if time.time() - start_time > DEADLINE:
            break
        placed, vol = greedy_pack(order)
        placed, vol = reinsert(placed, vol)
        result, vol = to_result(placed)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    # Extra random restarts with time to spare
    budget_left = DEADLINE - (time.time() - start_time)
    extra = 0
    while budget_left > 8.0 and extra < 40:
        if time.time() - start_time > DEADLINE:
            break
        o = list(range(n))
        rng.shuffle(o)
        # bias: occasionally mix a sorted prefix with shuffle
        if rng.random() < 0.5:
            k = rng.randint(0, n)
            o[:k] = sorted(o[:k], key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
        placed, vol = greedy_pack(o)
        placed, vol = reinsert(placed, vol)
        result, vol = to_result(placed)
        if vol > best_vol:
            best_vol = vol
            best_result = result
        extra += 1
        budget_left = DEADLINE - (time.time() - start_time)

    if best_result is None:
        # Fallback: place greedily by volume
        placed, vol = greedy_pack(sorted(range(n), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2]))
        best_result, _ = to_result(placed)

    return best_result
