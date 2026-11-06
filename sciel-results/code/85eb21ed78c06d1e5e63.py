import time
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 145

    box_dims = [(w, h, d) for w, h, d in boxes]
    volumes = np.array([w * h * d for w, h, d in boxes], dtype=np.float64)
    container_vol = float(W * H * D)
    if container_vol <= 0:
        return [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    def valid_place(bin_boxes, x, y, z, w, h, d):
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

    def candidate_positions(bin_boxes, w, h, d):
        pos_set = {(0, 0, 0)}
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            for dx in (0, pw):
                for dy in (0, ph):
                    for dz in (0, pd):
                        pos_set.add((px + dx, py + dy, pz + dz))
        cands = []
        for (cx, cy, cz) in pos_set:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                cands.append((cx, cy, cz))
        cands.sort(key=lambda p: (p[2], p[1], p[0]))
        return cands

    def place_box(bin_boxes, w, h, d):
        for (x, y, z) in candidate_positions(bin_boxes, w, h, d):
            if valid_place(bin_boxes, x, y, z, w, h, d):
                return (x, y, z)
        return None

    def build_packing(order):
        bins = []
        bin_vols = []
        placements = [None] * n
        for idx in order:
            w, h, d = box_dims[idx]
            chosen_bin = -1
            chosen_pos = None
            best_key = None
            for b in range(len(bins)):
                spot = place_box(bins[b], w, h, d)
                if spot is not None:
                    key = (-bin_vols[b], spot[2], spot[1], spot[0])
                    if best_key is None or key < best_key:
                        best_key = key
                        chosen_bin = b
                        chosen_pos = spot
            if chosen_bin >= 0:
                x, y, z = chosen_pos
                bins[chosen_bin].append((x, y, z, w, h, d))
                bin_vols[chosen_bin] += w * h * d
                placements[idx] = (chosen_bin, x, y, z, w, h, d)
            else:
                b = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_vols.append(w * h * d)
                placements[idx] = (b, 0, 0, 0, w, h, d)
        return placements, bins

    def obj_value(placements):
        if not placements:
            return 0.0
        bset = set(p[0] for p in placements)
        B = len(bset)
        last = max(bset)
        last_vol = 0.0
        for p in placements:
            if p[0] == last:
                last_vol += p[4] * p[5] * p[6]
        return B + 0.1 * (1.0 - last_vol / container_vol)

    def extract_bin(placements, bid):
        return [p[1:7] for p in placements if p is not None and p[0] == bid]

    def remove_and_repack(placements, bid):
        movers = [i for i in range(n) if placements[i] is not None and placements[i][0] == bid]
        if not movers:
            return placements
        movers.sort(key=lambda i: -volumes[i])
        new_pl = list(placements)
        other_bins = {}
        for bid2 in set(p[0] for p in new_pl if p is not None):
            if bid2 != bid:
                other_bins[bid2] = extract_bin(new_pl, bid2)
        for i in movers:
            w, h, d = box_dims[i]
            placed = False
            for bid2 in sorted(other_bins):
                spot = place_box(other_bins[bid2], w, h, d)
                if spot is not None:
                    other_bins[bid2].append((spot[0], spot[1], spot[2], w, h, d))
                    new_pl[i] = (bid2, spot[0], spot[1], spot[2], w, h, d)
                    placed = True
                    break
            if not placed:
                return None
        return new_pl

    def move_one(placements, idx, target):
        w, h, d = box_dims[idx]
        tgt_boxes = extract_bin(placements, target)
        spot = place_box(tgt_boxes, w, h, d)
        if spot is None:
            return None
        new_pl = list(placements)
        new_pl[idx] = (target, spot[0], spot[1], spot[2], w, h, d)
        return new_pl

    def check_overlap(placements):
        bin_map = defaultdict(list)
        for i, p in enumerate(placements):
            if p is not None:
                bin_map[p[0]].append((i, p))
        for blist in bin_map.values():
            for a in range(len(blist)):
                ia, pa = blist[a]
                for b in range(a + 1, len(blist)):
                    ib, pb = blist[b]
                    if (pa[1] < pb[1] + pb[4] and pa[1] + pa[4] > pb[1] and
                        pa[2] < pb[2] + pb[5] and pa[2] + pa[5] > pb[2] and
                        pa[3] < pb[3] + pb[6] and pa[3] + pa[6] > pb[3]):
                        return False
        return True

    def valid_solution(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            if p[0] < 0 or p[1] < 0 or p[2] < 0 or p[3] < 0:
                return False
            if p[1] + p[4] > W or p[2] + p[5] > H or p[3] + p[6] > D:
                return False
        return check_overlap(placements)

    # Backtracking search for minimum bins
    def backtrack_min_bins(order, max_bins):
        """Try to pack all boxes into at most max_bins bins using backtracking."""
        bins = [[] for _ in range(max_bins)]
        
        def solve(pos):
            if pos == n:
                return True
            if time.time() > deadline - 5:
                return False
            idx = order[pos]
            w, h, d = box_dims[idx]
            for b in range(max_bins):
                for (x, y, z) in candidate_positions(bins[b], w, h, d):
                    if valid_place(bins[b], x, y, z, w, h, d):
                        bins[b].append((x, y, z, w, h, d))
                        if solve(pos + 1):
                            return True
                        bins[b].pop()
            return False
        
        return solve(0)

    def build_from_backtrack(order, max_bins):
        """Build placements from a successful backtracking search."""
        bins = [[] for _ in range(max_bins)]
        
        def solve(pos):
            if pos == n:
                return True
            if time.time() > deadline - 5:
                return False
            idx = order[pos]
            w, h, d = box_dims[idx]
            for b in range(max_bins):
                for (x, y, z) in candidate_positions(bins[b], w, h, d):
                    if valid_place(bins[b], x, y, z, w, h, d):
                        bins[b].append((x, y, z, w, h, d))
                        if solve(pos + 1):
                            return True
                        bins[b].pop()
            return False
        
        if not solve(0):
            return None
        
        placements = [None] * n
        for b in range(max_bins):
            for (x, y, z, w, h, d) in bins[b]:
                # Find which box this is
                for i in range(n):
                    if placements[i] is None and box_dims[i] == (w, h, d):
                        placements[i] = (b, x, y, z, w, h, d)
                        break
        return placements

    best_pl = None
    best_obj = float('inf')

    vol_order = list(np.argsort(-volumes))
    max_order = list(np.argsort(-np.array([max(w, h, d) for w, h, d in box_dims])))
    min_order = list(np.argsort(-np.array([min(w, h, d) for w, h, d in box_dims])))
    surf_order = list(np.argsort(-np.array([2*(w*h + w*d + h*d) for w, h, d in box_dims])))

    random.seed(7)
    orders = [vol_order, max_order, min_order, surf_order]
    for k in range(15):
        base = np.array(vol_order, dtype=np.int64)
        perm = np.random.permutation(n)
        base = base[perm]
        orders.append(list(base))

    # First, try greedy packing with multiple orders
    for order in orders:
        if time.time() > deadline - 60:
            break
        pl, _ = build_packing(order)
        s = obj_value(pl)
        if s < best_obj:
            best_obj = s
            best_pl = pl

    # Then, try backtracking to find minimum bins
    if time.time() < deadline - 30:
        # Estimate minimum bins needed
        total_vol = sum(volumes)
        min_bins_estimate = max(1, int(np.ceil(total_vol / container_vol)))
        
        # Try from estimated minimum up
        for mb in range(min_bins_estimate, min_bins_estimate + 4):
            if time.time() > deadline - 10:
                break
            success = False
            for order in [vol_order, max_order]:
                if time.time() > deadline - 10:
                    break
                bins = [[] for _ in range(mb)]
                
                def solve_bt(pos, bins_local, order_local, mb_local):
                    if pos == n:
                        return True
                    if time.time() > deadline - 10:
                        return False
                    idx = order_local[pos]
                    w, h, d = box_dims[idx]
                    for b in range(mb_local):
                        for (x, y, z) in candidate_positions(bins_local[b], w, h, d):
                            if valid_place(bins_local[b], x, y, z, w, h, d):
                                bins_local[b].append((x, y, z, w, h, d))
                                if solve_bt(pos + 1, bins_local, order_local, mb_local):
                                    return True
                                bins_local[b].pop()
                    return False
                
                if solve_bt(0, bins, order, mb):
                    # Build placements from successful bins
                    placements = [None] * n
                    for b in range(mb):
                        for (x, y, z, w, h, d) in bins[b]:
                            for i in range(n):
                                if placements[i] is None and box_dims[i] == (w, h, d):
                                    placements[i] = (b, x, y, z, w, h, d)
                                    break
                    if all(p is not None for p in placements):
                        s = obj_value(placements)
                        if s < best_obj:
                            best_obj = s
                            best_pl = placements
                    success = True
                    break
            if success:
                break

    def local_search(pl, score, max_iter=200):
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            improved = False
            bset = sorted(set(p[0] for p in pl))
            if not bset:
                break
            last = max(bset)
            movers = [i for i in range(n) if pl[i][0] == last]
            # Try to consolidate the last bin by moving its boxes to other bins
            if len(movers) <= 25:
                new_pl = remove_and_repack(pl, last)
                if new_pl is not None:
                    ns = obj_value(new_pl)
                    if ns < score - 1e-12:
                        pl = new_pl
                        score = ns
                        improved = True
            # Try moving individual boxes to reduce bin count or improve last-bin fill
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                cur_bin = pl[i][0]
                for tb in bset:
                    if tb == cur_bin:
                        continue
                    new_pl = move_one(pl, i, tb)
                    if new_pl is not None:
                        ns = obj_value(new_pl)
                        if ns < score - 1e-12:
                            pl = new_pl
                            score = ns
                            improved = True
                            bset = sorted(set(p[0] for p in pl))
                            break
                if not improved:
                    continue
            if not improved:
                break
        return pl, score

    if best_pl is not None:
        best_pl, best_obj = local_search(best_pl, best_obj, max_iter=400)

    if not valid_solution(best_pl):
        # Fallback: simple shelf packing
        b, x, y, z = 0, 0, 0, 0
        row_h = 0
        layer_d = 0
        best_pl = []
        for w, h, d in box_dims:
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                x = 0
                row_h = 0
                layer_d = 0
            if z + d > D:
                b += 1
                x, y, z = 0, 0, 0
                row_h = 0
                layer_d = 0
            best_pl.append((b, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)

    return best_pl
