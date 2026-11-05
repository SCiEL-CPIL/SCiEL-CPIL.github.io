import itertools
import time
from itertools import permutations

# EVOLVE-BLOCK-START
def _rotations(dims):
    a, b, c = dims
    seen = set()
    out = []
    for p in permutations((a, b, c)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _fits(p, W, H, D):
    return p[0] <= W and p[1] <= H and p[2] <= D

def _shelves_ok(shelves, W, H, D):
    for rows in shelves:
        for (w, h, d) in rows:
            if w > W or h > H or d > D:
                return False
    return True

def _pack_shelves(shelves, boxes, W, H, D):
    """Greedy first-fit-decreasing (with rotations) into existing shelves."""
    n = len(boxes)
    idx = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    placed = [False]*n
    for i in idx:
        w, h, d = boxes[i]
        rots = [r for r in _rotations((w, h, d)) if _fits(r, W, H, D)]
        best = None
        for (rw, rh, rd) in rots:
            for si, rows in enumerate(shelves):
                for ri, row in enumerate(rows):
                    if rh > row[0][1]:
                        continue
                    if row[0][0] + rw <= W and row[0][2] + rd <= D:
                        if best is None or si < best[0]:
                            best = (si, ri, rw, rh, rd)
                if best is not None:
                    break
            if best is not None:
                break
        if best is not None:
            si, ri, rw, rh, rd = best
            shelves[si][ri].append((rw, rh, rd))
            placed[i] = True
        else:
            # open a new row in a new layer
            if rh <= H and rd <= D and rw <= W:
                shelves.append([[ (rw, rh, rd) ]])
                placed[i] = True
    return placed

def _shelves_volume(shelves):
    v = 0
    for rows in shelves:
        for (w, h, d) in rows:
            v += w*h*d
    return v

def _solve(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0 or W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Precompute feasible rotations per box
    rots = []
    for i in range(n):
        w, h, d = boxes[i]
        rots.append([r for r in _rotations((w, h, d)) if _fits(r, W, H, D)])

    # order by volume desc
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    # ---- Phase 1: exhaustive spine search over the first k largest boxes ----
    k = min(n, 4)
    spine_idx = order[:k]
    spine_rots = [rots[i] for i in spine_idx]

    best_shelves = None
    best_vol = -1
    start = time.time()
    deadline = start + 90.0  # leave margin

    # enumerate rotation combos for spine boxes
    spine_combos = [list(itertools.product(*spine_rots))] if all(spine_rots) else []
    if not spine_combos:
        spine_combos = [tuple(None for _ in spine_idx)]

    def _build_shelves(spine_assign):
        """Place spine boxes (fixed rotations) into shelves greedily, then fill rest."""
        shelves = []
        # place spine boxes one by one (best-fit)
        for bi, rot in enumerate(spine_assign):
            if rot is None:
                continue
            rw, rh, rd = rot
            placed_here = False
            for si, rows in enumerate(shelves):
                for ri, row in enumerate(rows):
                    if rh > row[0][1]:
                        continue
                    if row[0][0] + rw <= W and row[0][2] + rd <= D:
                        shelves[si][ri].append((rw, rh, rd))
                        placed_here = True
                        break
                if placed_here:
                    break
            if not placed_here:
                if rw <= W and rh <= H and rd <= D:
                    shelves.append([[ (rw, rh, rd) ]])
        # fill remaining boxes greedily
        remaining = [i for i in order if i not in spine_idx]
        for i in remaining:
            w, h, d = boxes[i]
            best = None
            for (rw, rh, rd) in rots[i]:
                for si, rows in enumerate(shelves):
                    for ri, row in enumerate(rows):
                        if rh > row[0][1]:
                            continue
                        if row[0][0] + rw <= W and row[0][2] + rd <= D:
                            if best is None or si < best[0]:
                                best = (si, ri, rw, rh, rd)
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is not None:
                si, ri, rw, rh, rd = best
                shelves[si][ri].append((rw, rh, rd))
            else:
                if rw <= W and rh <= H and rd <= D:
                    shelves.append([[ (rw, rh, rd) ]])
        return shelves

    # search spine rotation combos (bounded)
    combos = list(itertools.product(*[r if r else [(None,)] for r in spine_rots]))
    max_combos = 4096
    for c in combos[:max_combos]:
        if time.time() > deadline:
            break
        shelves = _build_shelves(c)
        v = _shelves_volume(shelves)
        if v > best_vol:
            best_vol = v
            best_shelves = shelves

    # ---- Phase 2: greedy from scratch (no fixed spine) as a baseline ----
    shelves2 = []
    for i in order:
        w, h, d = boxes[i]
        best = None
        for (rw, rh, rd) in rots[i]:
            for si, rows in enumerate(shelves2):
                for ri, row in enumerate(rows):
                    if rh > row[0][1]:
                        continue
                    if row[0][0] + rw <= W and row[0][2] + rd <= D:
                        if best is None or si < best[0]:
                            best = (si, ri, rw, rh, rd)
                if best is not None:
                    break
            if best is not None:
                break
        if best is not None:
            si, ri, rw, rh, rd = best
            shelves2[si][ri].append((rw, rh, rd))
        else:
            if rw <= W and rh <= H and rd <= D:
                shelves2.append([[ (rw, rh, rd) ]])
    v2 = _shelves_volume(shelves2)
    if v2 > best_vol:
        best_vol = v2
        best_shelves = shelves2

    if best_shelves is None:
        best_shelves = []

    # ---- Phase 3: build the final placement list from best_shelves ----
    # We need to map back to original box indices. Since we only stored dims, we
    # re-run the same greedy deterministically to produce per-box placements.
    # To keep it simple and correct, we rebuild placements by re-deriving which
    # box went where using the same ordering and matching dims greedily.
    # Simpler: just re-run the greedy that produced best_shelves and record boxes.

    # Reconstruct by replaying: we stored only dims, so we re-run the exact same
    # algorithm but this time recording the box index for each placed cell.
    def _replay(spine_assign):
        shelves = []
        box_at = {}  # (si, ri, pos) -> box index
        for bi, rot in enumerate(spine_assign):
            if rot is None:
                continue
            rw, rh, rd = rot
            placed_here = False
            for si, rows in enumerate(shelves):
                for ri, row in enumerate(rows):
                    if rh > row[0][1]:
                        continue
                    if row[0][0] + rw <= W and row[0][2] + rd <= D:
                        shelves[si][ri].append((rw, rh, rd))
                        box_at[(si, ri, len(shelves[si][ri])-1)] = spine_idx[bi]
                        placed_here = True
                        break
                if placed_here:
                    break
            if not placed_here:
                if rw <= W and rh <= H and rd <= D:
                    shelves.append([[ (rw, rh, rd) ]])
                    box_at[(len(shelves)-1, 0, 0)] = spine_idx[bi]
        remaining = [i for i in order if i not in spine_idx]
        for i in remaining:
            w, h, d = boxes[i]
            best = None
            for (rw, rh, rd) in rots[i]:
                for si, rows in enumerate(shelves):
                    for ri, row in enumerate(rows):
                        if rh > row[0][1]:
                            continue
                        if row[0][0] + rw <= W and row[0][2] + rd <= D:
                            if best is None or si < best[0]:
                                best = (si, ri, rw, rh, rd)
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is not None:
                si, ri, rw, rh, rd = best
                shelves[si][ri].append((rw, rh, rd))
                box_at[(si, ri, len(shelves[si][ri])-1)] = i
            else:
                if rw <= W and rh <= H and rd <= D:
                    shelves.append([[ (rw, rh, rd) ]])
                    box_at[(len(shelves)-1, 0, 0)] = i
        return shelves, box_at

    # Determine which replay produced best_shelves
    # We'll just replay the greedy baseline (no spine) and the best spine combo;
    # pick whichever matches best_vol, else use baseline.
    # For robustness, replay baseline first.
    shelves_r, box_at_r = _replay([None]*len(spine_idx))
    v_r = _shelves_volume(shelves_r)
    if v_r >= best_vol - 1e-9:
        best_shelves = shelves_r
        box_at = box_at_r
    else:
        # replay best spine combo (first one that matched)
        # re-find it
        found = False
        for c in combos[:max_combos]:
            shelves_c, box_at_c = _replay(c)
            if _shelves_volume(shelves_c) >= best_vol - 1e-9:
                best_shelves = shelves_c
                box_at = box_at_c
                found = True
                break
        if not found:
            best_shelves = shelves_r
            box_at = box_at_r

    # ---- Emit placements ----
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    for si, rows in enumerate(best_shelves):
        z0 = 0
        for ri, row in enumerate(rows):
            # row height = max h in row; depth = max d in row
            rh = max(r[1] for r in row)
            rd = max(r[2] for r in row)
            x0 = 0
            for pos, (rw, rhb, rdb) in enumerate(row):
                bi = box_at.get((si, ri, pos))
                if bi is not None:
                    # place at (x0, 0, z0) with dims (rw, rhb, rdb)
                    placements[bi] = (0, x0, 0, z0, rw, rhb, rdb)
                    x0 += rw
            z0 += rd
    return placements

def load(boxes, container):
    start = time.time()
    deadline = start + 105.0  # safety margin under 120s
    try:
        res = _solve(boxes, container)
        if time.time() > deadline:
            # fallback: trivial shelf (no rotation) to guarantee return
            return _fallback(boxes, container)
        return res
    except Exception:
        return _fallback(boxes, container)

def _fallback(boxes, container):
    W, H, D = container
    placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + d > D:
            placements.append((-1, 0, 0, 0, w, h, d))
            continue
        placements.append((0, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements
# EVOLVE-BLOCK-END
