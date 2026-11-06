"""3D Bin Packing: place boxes unrotated into numbered bins, minimize bins + last-bin fill penalty."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(0, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Index boxes by original index to preserve return order
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    def volume(w, h, d):
        return w * h * d

    def can_place(bin_items, w, h, d, x, y, z):
        """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_items:
            if (x < bx + bw and x + w > bx) and \
               (y < by + bh and y + h > by) and \
               (z < bz + bd and z + d > bz):
                return False
        return True

    def find_spot(bin_items, w, h, d):
        """Find a valid (x,y,z) for box in this bin using candidate positions from box edges."""
        xs = {0}
        ys = {0}
        zs = {0}
        for (bx, by, bz, bw, bh, bd) in bin_items:
            if bx + bw <= W:
                xs.add(bx + bw)
            if by + bh <= H:
                ys.add(by + bh)
            if bz + bd <= D:
                zs.add(bz + bd)

        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)

        best_pos = None
        best_score = float('inf')
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_items, w, h, d, x, y, z):
                        # Prefer positions that minimize wasted space
                        score = x + y * (W + 1) + z * (W + 1) * (H + 1)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        return best_pos

    def try_order(order):
        """Try to pack boxes in given order using first-fit-decreasing with best-fit placement."""
        bins = []
        placements = {}

        for idx, w, h, d in order:
            placed = False
            best_bin = -1
            best_pos = None
            best_waste = float('inf')

            for bi in range(len(bins)):
                pos = find_spot(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    waste = x + y + z
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = bi
                        best_pos = pos

            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements[idx] = (best_bin, x, y, z, w, h, d)
            else:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (bi, 0, 0, 0, w, h, d)

        return placements

    def score_placements(placements):
        """Compute score: B + 0.1*(1 - last_bin_fill_fraction)"""
        if not placements:
            return 0.0
        bin_ids = set(p[0] for p in placements.values())
        B = len(bin_ids)
        last_bin = max(bin_ids)
        last_vol = 0
        for p in placements.values():
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
        container_vol = W * H * D
        if container_vol == 0:
            fill_frac = 1.0
        else:
            fill_frac = last_vol / container_vol
        return B + 0.1 * (1 - fill_frac)

    def to_output(placements, n):
        """Convert placements dict to list in original box order."""
        result = []
        for i in range(n):
            result.append(placements[i])
        return result

    def local_search(placements, bins, order, max_iters=150):
        """Local search: try swapping boxes between bins to improve score."""
        current_score = score_placements(placements)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            box_list = list(order)
            for i in range(len(box_list)):
                if time.time() > deadline - 3:
                    return placements, bins, current_score
                for j in range(i + 1, len(box_list)):
                    idx_i, wi, hi, di = box_list[i]
                    idx_j, wj, hj, dj = box_list[j]
                    bin_i = placements[idx_i][0]
                    bin_j = placements[idx_j][0]
                    if bin_i == bin_j:
                        continue

                    orig_pos_i = placements[idx_i]
                    orig_pos_j = placements[idx_j]

                    # Find and remove box_i from bin_i
                    bin_i_items = bins[bin_i]
                    box_i_data = None
                    for k, item in enumerate(bin_i_items):
                        if item[0] == orig_pos_i[1] and item[1] == orig_pos_i[2] and item[2] == orig_pos_i[3] and \
                           item[3] == wi and item[4] == hi and item[5] == di:
                            box_i_data = bin_i_items.pop(k)
                            break
                    if box_i_data is None:
                        continue

                    # Find and remove box_j from bin_j
                    bin_j_items = bins[bin_j]
                    box_j_data = None
                    for k, item in enumerate(bin_j_items):
                        if item[0] == orig_pos_j[1] and item[1] == orig_pos_j[2] and item[2] == orig_pos_j[3] and \
                           item[3] == wj and item[4] == hj and item[5] == dj:
                            box_j_data = bin_j_items.pop(k)
                            break
                    if box_j_data is None:
                        bins[bin_i].append(box_i_data)
                        continue

                    # Try placing box_i in bin_j
                    pos_i_new = find_spot(bin_j_items, wi, hi, di)
                    if pos_i_new is None:
                        bins[bin_i].append(box_i_data)
                        bins[bin_j].append(box_j_data)
                        continue

                    # Try placing box_j in bin_i
                    pos_j_new = find_spot(bin_i_items, wj, hj, dj)
                    if pos_j_new is None:
                        bins[bin_i].append(box_i_data)
                        bins[bin_j].append(box_j_data)
                        continue

                    # Accept the swap tentatively
                    x_i, y_i, z_i = pos_i_new
                    x_j, y_j, z_j = pos_j_new
                    bin_j_items.append((x_i, y_i, z_i, wi, hi, di))
                    bin_i_items.append((x_j, y_j, z_j, wj, hj, dj))

                    placements[idx_i] = (bin_j, x_i, y_i, z_i, wi, hi, di)
                    placements[idx_j] = (bin_i, x_j, y_j, z_j, wj, hj, dj)

                    new_score = score_placements(placements)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        bin_j_items.remove((x_i, y_i, z_i, wi, hi, di))
                        bin_i_items.remove((x_j, y_j, z_j, wj, hj, dj))
                        placements[idx_i] = orig_pos_i
                        placements[idx_j] = orig_pos_j
                        bin_i_items.append(box_i_data)
                        bin_j_items.append(box_j_data)

        return placements, bins, current_score

    best_placements = None
    best_score_val = float('inf')
    best_bins = None
    best_order = None

    # Multiple sorting strategies
    strategies = [
        lambda items: sorted(items, key=lambda x: -volume(x[1], x[2], x[3])),
        lambda items: sorted(items, key=lambda x: -(max(x[1], x[2], x[3]))),
        lambda items: sorted(items, key=lambda x: -(x[1] + x[2] + x[3])),
        lambda items: sorted(items, key=lambda x: -(x[1] * x[2], x[3])),
        lambda items: sorted(items, key=lambda x: -(x[2] * x[3], x[1])),
        lambda items: sorted(items, key=lambda x: -(x[1] * x[3], x[2])),
        lambda items: sorted(items, key=lambda x: (-volume(x[1], x[2], x[3]), -max(x[1], x[2], x[3]))),
        lambda items: sorted(items, key=lambda x: (-max(x[1], x[2], x[3]), -volume(x[1], x[2], x[3]))),
    ]

    for si, strat in enumerate(strategies):
        if time.time() > deadline - 15:
            break
        try:
            order = strat(indexed)
            placements = try_order(order)
            if placements is None or len(placements) != n:
                continue
            s = score_placements(placements)
            if s < best_score_val:
                best_score_val = s
                best_placements = placements
                best_order = order
                bin_dict = {}
                for idx, w, h, d in order:
                    p = placements[idx]
                    bi = p[0]
                    if bi not in bin_dict:
                        bin_dict[bi] = []
                    bin_dict[bi].append((p[1], p[2], p[3], w, h, d))
                best_bins = bin_dict
        except Exception:
            continue

    # Local search on best result
    if best_placements is not None and time.time() < deadline - 20:
        bins_list = [best_bins[k] for k in sorted(best_bins.keys())]
        if time.time() < deadline - 5:
            best_placements, _, best_score_val = local_search(best_placements, bins_list, best_order, max_iters=100)

    # Random restarts if time remains
    if best_placements is not None and time.time() < deadline - 20:
        base_order = strategies[0](indexed)
        n_restarts = 0
        while time.time() < deadline - 5 and n_restarts < 50:
            n_restarts += 1
            shuffled = base_order[:]
            random.shuffle(shuffled)
            try:
                placements = try_order(shuffled)
                if placements is None or len(placements) != n:
                    continue
                s = score_placements(placements)
                if s < best_score_val:
                    best_score_val = s
                    best_placements = placements
                    best_order = shuffled
                    bin_dict = {}
                    for idx, w, h, d in shuffled:
                        p = placements[idx]
                        bi = p[0]
                        if bi not in bin_dict:
                            bin_dict[bi] = []
                        bin_dict[bi].append((p[1], p[2], p[3], w, h, d))
                    best_bins = bin_dict
            except Exception:
                continue

    # Fallback: simple shelf if everything failed
    if best_placements is None:
        best_placements = {}
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements[i] = (b, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return to_output(best_placements, n)

# EVOLVE-BLOCK-END
