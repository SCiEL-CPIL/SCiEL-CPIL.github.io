import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    volumes = np.array([w * h * d for w, h, d in boxes], dtype=np.float64)
    total_volume = float(volumes.sum())
    container_vol = float(W * H * D)
    if container_vol <= 0:
        return [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Sort indices by volume descending
    order_vol = list(np.argsort(-volumes))

    def is_valid_place(bin_boxes, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or x >= px + pw):
                if not (y + h <= py or y >= py + ph):
                    if not (z + d <= pz or z >= pz + pd):
                        return False
        return True

    def get_candidates(bin_boxes, w, h, d):
        """Generate candidate positions: origin + offsets from existing boxes."""
        cands = [(0, 0, 0)]
        seen = set(cands)
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            candidates = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]
            for c in candidates:
                if c not in seen and c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                    seen.add(c)
                    cands.append(c)
        return cands

    def find_spot(bin_boxes, w, h, d):
        """Find the best valid spot, preferring (z, y, x) minimization."""
        cands = get_candidates(bin_boxes, w, h, d)
        best = None
        best_key = None
        for (x, y, z) in cands:
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (x, y, z)
        return best

    def find_spot_min_waste(bin_boxes, w, h, d):
        """Find valid spot minimizing wasted space (z*y*x)."""
        cands = get_candidates(bin_boxes, w, h, d)
        best = None
        best_key = None
        for (x, y, z) in cands:
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (x, y, z)
        return best

    def compute_score(placements):
        bin_ids = set()
        for p in placements:
            bin_ids.add(p[0])
        B = len(bin_ids)
        last_bin = max(bin_ids)
        last_bin_vol = 0.0
        for p in placements:
            if p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        return B + 0.1 * (1.0 - last_bin_vol / container_vol)

    def initial_packing(order):
        """Best Fit Decreasing with given order."""
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = [None] * n

        for idx in order:
            w, h, d = boxes[idx]
            best_bin = -1
            best_spot = None
            best_key = None

            for b_idx, bin_boxes in enumerate(bins):
                spot = find_spot(bin_boxes, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    bin_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_boxes)
                    key = (-bin_vol, z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_bin = b_idx
                        best_spot = spot

            if best_bin >= 0:
                x, y, z = best_spot
                bins[best_bin].append((x, y, z, w, h, d))
                placements[idx] = (best_bin, x, y, z, w, h, d)
            else:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (b_idx, 0, 0, 0, w, h, d)

        return placements

    def get_bin_boxes(placements, bin_id):
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]

    def try_move_box(placements, box_idx, target_bin_id):
        """Try to move box_idx to target_bin_id. Returns new_placements or None."""
        w, h, d = placements[box_idx][4], placements[box_idx][5], placements[box_idx][6]
        target_boxes = get_bin_boxes(placements, target_bin_id)
        spot = find_spot(target_boxes, w, h, d)
        if spot is None:
            return None
        new_placements = list(placements)
        new_placements[box_idx] = (target_bin_id, spot[0], spot[1], spot[2], w, h, d)
        return new_placements

    def try_consolidate_bin(placements, bin_to_remove):
        """Try to move all boxes from bin_to_remove into other bins."""
        boxes_in_bin = [i for i in range(n) if placements[i] is not None and placements[i][0] == bin_to_remove]
        if not boxes_in_bin:
            return placements

        boxes_in_bin.sort(key=lambda i: -volumes[i])

        work_placements = list(placements)
        work_bins = {}
        for b_id in set(p[0] for p in work_placements):
            if b_id != bin_to_remove:
                work_bins[b_id] = get_bin_boxes(work_placements, b_id)

        for box_idx in boxes_in_bin:
            w, h, d = boxes[box_idx]
            placed = False
            for b_id, b_boxes in sorted(work_bins.items()):
                spot = find_spot(b_boxes, w, h, d)
                if spot is not None:
                    work_bins[b_id].append((spot[0], spot[1], spot[2], w, h, d))
                    work_placements[box_idx] = (b_id, spot[0], spot[1], spot[2], w, h, d)
                    placed = True
                    break
            if not placed:
                return None

        return work_placements

    def local_search(placements, current_score, max_passes=100):
        """Iterative local search to improve the packing."""
        improved = True
        pass_count = 0

        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            if time.time() > deadline - 5:
                break

            last_bin_id = max(p[0] for p in placements)
            all_bin_ids = sorted(set(p[0] for p in placements))

            # Phase A: Try to move boxes from other bins into the last bin
            for i in order_vol:
                if time.time() > deadline - 5:
                    break
                if placements[i] is None:
                    continue
                orig_bin = placements[i][0]
                if orig_bin == last_bin_id:
                    continue

                w, h, d = boxes[i]
                new_pl = try_move_box(placements, i, last_bin_id)
                if new_pl is not None:
                    new_score = compute_score(new_pl)
                    if new_score < current_score - 1e-12:
                        placements = new_pl
                        current_score = new_score
                        improved = True
                        last_bin_id = max(p[0] for p in placements)

            # Phase B: Try to eliminate the last bin
            if time.time() > deadline - 5:
                break

            last_bin_id = max(p[0] for p in placements)
            last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]

            if len(last_bin_boxes) <= 6:
                new_pl = try_consolidate_bin(placements, last_bin_id)
                if new_pl is not None:
                    new_score = compute_score(new_pl)
                    if new_score < current_score - 1e-12:
                        placements = new_pl
                        current_score = new_score
                        improved = True

            # Phase C: Try to reduce bin count by moving boxes between bins
            if time.time() > deadline - 10:
                break

            all_bin_ids = sorted(set(p[0] for p in placements))
            if len(all_bin_ids) > 1:
                for i in order_vol:
                    if time.time() > deadline - 10:
                        break
                    if placements[i] is None:
                        continue
                    src_bin = placements[i][0]
                    w, h, d = boxes[i]

                    for tgt_bin in all_bin_ids:
                        if tgt_bin == src_bin:
                            continue
                        new_pl = try_move_box(placements, i, tgt_bin)
                        if new_pl is not None:
                            new_score = compute_score(new_pl)
                            if new_score < current_score - 1e-12:
                                placements = new_pl
                                current_score = new_score
                                improved = True
                                all_bin_ids = sorted(set(p[0] for p in placements))
                                break

        return placements, current_score

    def is_valid(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True

    # ---- Multiple diverse initial packings ----
    best_placements = None
    best_score = float('inf')

    # Strategy 1: Volume descending
    pl1 = initial_packing(order_vol)
    s1 = compute_score(pl1)
    if s1 < best_score:
        best_score = s1
        best_placements = pl1

    # Strategy 2: Sort by longest side descending
    order_long = list(np.argsort(-np.array([max(w, h, d) for w, h, d in boxes])))
    if time.time() < deadline - 20:
        pl2 = initial_packing(order_long)
        s2 = compute_score(pl2)
        if s2 < best_score:
            best_score = s2
            best_placements = pl2

    # Strategy 3: Sort by (w*h*d) with tie-break on max dimension
    order_vol_max = list(np.argsort(-np.array([w * h * d for w, h, d in boxes]) - 0.01 * np.array([max(w, h, d) for w, h, d in boxes])))
    if time.time() < deadline - 20:
        pl3 = initial_packing(order_vol_max)
        s3 = compute_score(pl3)
        if s3 < best_score:
            best_score = s3
            best_placements = pl3

    # Strategy 4: Random shuffle with volume bias
    if time.time() < deadline - 30:
        random.seed(42)
        for _ in range(5):
            shuffled = list(range(n))
            random.shuffle(shuffled)
            # Sort by volume but with some randomness
            order_rand = sorted(shuffled, key=lambda i: -volumes[i] * random.uniform(0.8, 1.2))
            plr = initial_packing(order_rand)
            sr = compute_score(plr)
            if sr < best_score:
                best_score = sr
                best_placements = plr

    # ---- Local search on best initial solution ----
    if best_placements is not None and time.time() < deadline - 10:
        best_placements, best_score = local_search(best_placements, best_score)

    # ---- Try local search on other good initial solutions too ----
    if time.time() < deadline - 15:
        for order in [order_vol, order_long]:
            if time.time() > deadline - 15:
                break
            pl = initial_packing(order)
            s = compute_score(pl)
            if s < best_score + 0.5:  # Only if close to best
                pl, s = local_search(pl, s, max_passes=50)
                if s < best_score:
                    best_score = s
                    best_placements = pl

    # ---- Final verification and fallback ----
    if not is_valid(best_placements):
        # Fallback: Simple shelf packing
        best_placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return best_placements
