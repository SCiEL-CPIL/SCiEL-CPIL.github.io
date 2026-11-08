"""Combined 3D bin packer: efficient best-fit with multi-strategy sorting
and time-bounded relocation to minimize bin count."""
import time
import random

_DEADLINE = 140.0  # safety margin before 150s kill


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    cvol = W * H * D
    t0 = time.time()

    def compute_score(placements):
        if not placements:
            return 0.0
        bins_used = set()
        last_bin = -1
        for (b, x, y, z, w, h, d) in placements:
            bins_used.add(b)
            if b > last_bin:
                last_bin = b
        last_vol = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == last_bin:
                last_vol += w * h * d
        B = len(bins_used)
        return B + 0.1 * (1.0 - last_vol / cvol)

    def find_placement(placed, w, h, d):
        """Find best (x,y,z) for box in a bin. Returns (x,y,z) or None."""
        if w > W or h > H or d > D:
            return None

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

        xs = sorted(x for x in xs if x + w <= W)
        ys = sorted(y for y in ys if y + h <= H)
        zs = sorted(z for z in zs if z + d <= D)

        if not xs or not ys or not zs:
            return None

        # Iterate in (z, y, x) order for tightest packing
        for cz in zs:
            for cy in ys:
                for cx in xs:
                    ok = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + w <= px or px + pw <= cx or
                                cy + h <= py or py + ph <= cy or
                                cz + d <= pz or pz + pd <= cz):
                            ok = False
                            break
                    if ok:
                        return (cx, cy, cz)
        return None

    def solve(sorted_boxes):
        """Best-fit decreasing: try most-filled bins first."""
        bins = []  # list of (placed_list, vol)
        placements = [None] * n

        for orig_idx, w, h, d, vol in sorted_boxes:
            if time.time() > _DEADLINE:
                return None

            placed_in_bin = False
            if bins:
                # Sort bins by volume descending (best-fit)
                bin_order = sorted(range(len(bins)), key=lambda bi: -bins[bi][1])
                for bi in bin_order:
                    pos = find_placement(bins[bi][0], w, h, d)
                    if pos is not None:
                        x, y, z = pos
                        bins[bi][0].append((x, y, z, w, h, d))
                        bins[bi][1] += vol
                        placements[orig_idx] = (bi, x, y, z, w, h, d)
                        placed_in_bin = True
                        break

            if not placed_in_bin:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d), vol])
                placements[orig_idx] = (bi, 0, 0, 0, w, h, d)

        return placements

    # Indexed boxes
    indexed = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]

    strategies = [
        lambda bl: sorted(bl, key=lambda x: -x[4]),
        lambda bl: sorted(bl, key=lambda x: (-(x[1] + x[2] + x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[1] * x[2]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[2] * x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[1] * x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-max(x[1], x[2], x[3]), -x[4])),
    ]

    best_placements = None
    best_score = float('inf')

    for strat in strategies:
        if time.time() > _DEADLINE - 1.0:
            break
        sorted_boxes = strat(indexed)
        placements = solve(sorted_boxes)
        if placements is None:
            continue
        # Check all placed
        if any(p is None for p in placements):
            continue
        s = compute_score(placements)
        if s < best_score:
            best_score = s
            best_placements = list(placements)

    if best_placements is None:
        # Fallback: simple sequential
        best_placements = []
        b = 0
        x = y = z = 0
        row_h = layer_d = 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                layer_d = 0
                row_h = 0
            if z + d > D:
                b += 1
                x = y = z = 0
                row_h = layer_d = 0
            best_placements.append((b, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        best_score = compute_score(best_placements)

    # Relocation pass: try to move boxes from last bin to earlier bins
    if time.time() < _DEADLINE - 2.0 and best_placements:
        pos = list(best_placements)
        # Build bins from pos
        bins_map = {}
        for i, (bi, x, y, z, w, h, d) in enumerate(pos):
            if bi not in bins_map:
                bins_map[bi] = []
            bins_map[bi].append((x, y, z, w, h, d, i))

        used_bins = sorted(bins_map.keys())
        if len(used_bins) > 1:
            last_bin = used_bins[-1]
            attempts = 0
            while time.time() < _DEADLINE - 2.0 and attempts < 3000:
                attempts += 1
                last_bin = max(b for b, *_ in pos)
                used_bins = sorted(set(b for b, *_ in pos))
                if len(used_bins) <= 1:
                    break
                last_bin = used_bins[-1]
                cand = [i for i in range(n) if pos[i][0] == last_bin]
                random.shuffle(cand)
                moved = False
                for i in cand:
                    bi, x, y, z, w, h, d = pos[i]
                    # Remove from current bin
                    cur_bin = bins_map.get(bi, [])
                    cur_bin = [it for it in cur_bin if it[6] != i]
                    bins_map[bi] = cur_bin
                    # Try to place in each earlier bin
                    placed = False
                    for tb in used_bins[:-1]:
                        target_items = [(it[0], it[1], it[2], it[3], it[4], it[5]) for it in bins_map.get(tb, [])]
                        p = find_placement(target_items, w, h, d)
                        if p is not None:
                            nx, ny, nz = p
                            bins_map[tb].append((nx, ny, nz, w, h, d, i))
                            pos[i] = (tb, nx, ny, nz, w, h, d)
                            placed = True
                            break
                    if placed:
                        moved = True
                        break
                    else:
                        # Put back
                        bins_map[bi].append((x, y, z, w, h, d, i))
                        pos[i] = (bi, x, y, z, w, h, d)
                if not moved:
                    break

            s = compute_score(pos)
            if s < best_score:
                best_score = s
                best_placements = pos

    return best_placements
