"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    deadline = time.time() + 110.0  # explicit safety margin vs 150s kill

    # Precompute unique orientations (any axis rotation allowed)
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        orient.append(list(opts))

    def try_pack(order):
        """Fast 3D pack: keep list of placed boxes; for each candidate box and
        orientation, try to place at a set of anchor points (origin + 'top faces').
        Anchors are derived from placed boxes' top faces (x, y, z_top) and also
        the origin. We accept the first non-overlapping anchor (prefer lower z,
        then lower y, then lower x) to keep packs compact.
        """
        placed = []  # list of (x, y, z, w, h, d)
        filled = 0
        # Candidate anchor points: start with origin; grow as we place.
        anchors = [(0, 0, 0)]

        for idx in order:
            w, h, d = boxes[idx]
            best = None  # (z, y, x, x, y, z, ow, oh, od)
            for (ow, oh, od) in orient[idx]:
                # Quick bounds
                if ow > W or oh > H or od > D:
                    continue
                # Try anchors; limit anchor count for speed by checking all but
                # anchors list can grow; we cap by only using anchors within bounds.
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    # overlap check against placed
                    ok = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (ax + ow <= px or px + pw <= ax or
                                ay + oh <= py or py + ph <= ay or
                                az + od <= pz or pz + pd <= az):
                            ok = False
                            break
                    if ok:
                        key = (az, ay, ax, ax, ay, az, ow, oh, od)
                        if best is None or key < best:
                            best = key
                        break  # for this orientation, first valid anchor is fine
                        # (we still compare across orientations via key)
                if best is not None:
                    # don't break out of orientation loop; maybe another orientation
                    # yields a better (lower z) key
                    pass
            if best is None:
                continue  # leave out
            _, _, _, bx, by, bz, bw, bh, bd = best
            placed.append((bx, by, bz, bw, bh, bd))
            filled += bw * bh * bd
            # Add new anchor points: top face of this box
            new_anchor = (bx, by, bz + bd)
            # also side anchors to allow filling beside
            if bx + bw < W:
                new_anchor2 = (bx + bw, by, bz)
            else:
                new_anchor2 = None
            if by + bh < H:
                new_anchor3 = (bx, by + bh, bz)
            else:
                new_anchor3 = None
            anchors.append(new_anchor)
            if new_anchor2 is not None:
                anchors.append(new_anchor2)
            if new_anchor3 is not None:
                anchors.append(new_anchor3)
        return filled, placed

    # Build orderings
    def make_orders():
        orders = []
        vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        # 1) descending volume
        orders.append(sorted(range(n), key=lambda i: -vol[i]))
        # 2) ascending volume
        orders.append(sorted(range(n), key=lambda i: vol[i]))
        # 3) largest surface area first
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        # 4) largest min dimension first (tall/wide boxes first)
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        # 5) compactness: product of dims / (max dim) -> prefer squarish
        def compact(i):
            a, b, c = boxes[i]
            m = max(a, b, c)
            return (a * b * c) / m if m else 0
        orders.append(sorted(range(n), key=lambda i: -compact(i)))
        return orders

    best_filled = -1
    best_placed = []

    base_orders = make_orders()
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed = try_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed

    # Randomized restarts within remaining time: perturb orderings
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        # pick a random base ordering and do a random shuffle of small blocks
        base = rng.choice(base_orders)
        order = list(base)
        # random swaps
        for _ in range(rng.randint(0, 6)):
            i, j = rng.randrange(n), rng.randrange(n)
            order[i], order[j] = order[j], order[i]
        filled, placed = try_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
        attempts += 1

    # Build final placements
    placed_set = set()
    for (px, py, pz, pw, ph, pd) in best_placed:
        placed_set.add((px, py, pz, pw, ph, pd))

    # Map placed boxes back to original indices by matching dimensions + position
    # We need to know which original box each placed box corresponds to.
    # Reconstruct by re-running the winning order? Simpler: track during try_pack.
    # To keep it simple and correct, we re-derive by matching: for each placed box,
    # find an original box with matching (w,h,d) in some orientation that hasn't been used.
    used = [False] * n
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    # Greedy match placed -> original by exact dims
    for (px, py, pz, pw, ph, pd) in best_placed:
        target = (pw, ph, pd)
        found = -1
        for i in range(n):
            if used[i]:
                continue
            if target in orient[i]:
                found = i
                break
        if found != -1:
            used[found] = True
            placements[found] = (0, px, py, pz, pw, ph, pd)
        else:
            # fallback: find any unused with same volume
            for i in range(n):
                if used[i]:
                    continue
                a, b, c = boxes[i]
                if a * b * c == pw * ph * pd:
                    found = i
                    break
            if found != -1:
                used[found] = True
                placements[found] = (0, px, py, pz, pw, ph, pd)
    return placements
# EVOLVE-BLOCK-END
