"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    deadline = time.time() + 110.0  # safety margin vs 120s kill
    container_vol = W * H * D

    # Precompute unique orientations for each box
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((int(w), int(h), int(d))):
            opts.add(p)
        orient.append(list(opts))

    # Use numpy grid if container is small enough for fast overlap checks
    use_grid = (W * H * D) < 50_000_000
    if use_grid:
        grid = np.zeros((W, H, D), dtype=np.uint8)
    else:
        grid = None

    def try_pack(order):
        """Fast 3D pack: keep list of placed boxes; for each candidate box and
        orientation, try to place at a set of anchor points (origin + 'top faces').
        Anchors are derived from placed boxes' top faces (x, y, z_top) and also
        the origin. We accept the first non-overlapping anchor (prefer lower z,
        then lower y, then lower x) to keep packs compact.
        """
        if grid is not None:
            grid[:] = 0
        placed = []  # list of (x, y, z, w, h, d)
        filled = 0
        # Candidate anchor points: start with origin; grow as we place.
        anchors = [(0, 0, 0)]
        # Track used indices
        used_idx = set()
        mapping = {}  # idx -> (x, y, z, ow, oh, od)

        for idx in order:
            if idx in used_idx:
                continue
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
                    if grid is not None:
                        # Fast numpy check
                        if np.any(grid[ax:ax+ow, ay:ay+oh, az:az+od] > 0):
                            continue
                    else:
                        ok = True
                        for (px, py, pz, pw, ph, pd) in placed:
                            if not (ax + ow <= px or px + pw <= ax or
                                    ay + oh <= py or py + ph <= ay or
                                    az + od <= pz or pz + pd <= az):
                                ok = False
                                break
                        if not ok:
                            continue
                    key = (az, ay, ax, ax, ay, az, ow, oh, od)
                    if best is None or key < best:
                        best = key
                    break  # for this orientation, first valid anchor is fine
                # We don't break out of orientation loop; maybe another orientation
                # yields a better (lower z) key
            if best is None:
                continue  # leave out
            _, _, _, bx, by, bz, bw, bh, bd = best
            placed.append((bx, by, bz, bw, bh, bd))
            used_idx.add(idx)
            mapping[idx] = (bx, by, bz, bw, bh, bd)
            filled += bw * bh * bd
            if grid is not None:
                grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
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
        return filled, placed, used_idx, mapping

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
        # 6) largest max dimension first
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        return orders

    best_filled = -1
    best_placed = []
    best_used = set()
    best_mapping = {}

    base_orders = make_orders()
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed, used_idx, mapping = try_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_used = used_idx
            best_mapping = mapping

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
        filled, placed, used_idx, mapping = try_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_used = used_idx
            best_mapping = mapping
        attempts += 1

    # Build final placements
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Use the mapping from the best run to assign placements
    for idx, (px, py, pz, pw, ph, pd) in best_mapping.items():
        placements[idx] = (0, px, py, pz, pw, ph, pd)

    return placements
# EVOLVE-BLOCK-END
