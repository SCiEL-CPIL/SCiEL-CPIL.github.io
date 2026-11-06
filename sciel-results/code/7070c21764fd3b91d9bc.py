import time
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

    # Sort indices by volume descending for better initial packing
    order = list(np.argsort(-volumes))

    # Precompute candidate positions per box size
    # We'll generate candidates dynamically based on existing boxes' positions

    # ---- Occupancy grid helper ----
    # Using a 3D boolean array for O(1) collision checks when dimensions are small
    # But for large containers we'll use a set of occupied cells or just list-based
    # Given 40 boxes, list-based with early termination is fast enough
    
    def is_valid_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in a bin with given boxes.
        bin_boxes is list of (px,py,pz,pw,ph,pd)"""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Overlap check
            if not (x + w <= px or x >= px + pw):
                if not (y + h <= py or y >= py + ph):
                    if not (z + d <= pz or z >= pz + pd):
                        return False
        return True

    def get_candidate_positions(bin_boxes, w, h, d):
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
                if c not in seen:
                    seen.add(c)
                    cands.append(c)
        return cands

    def find_best_spot(bin_boxes, w, h, d):
        """Find the best valid spot, preferring (z, y, x) minimization."""
        cands = get_candidate_positions(bin_boxes, w, h, d)
        best = None
        best_key = None
        for (x, y, z) in cands:
            if x + w > W or y + h > H or z + d > D:
                continue
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (x, y, z)
        return best

    def find_spot_with_priority(bin_boxes, w, h, d, prefer_fill=True):
        """Find a valid spot. If prefer_fill, prefer positions that fill gaps (minimize z,y,x)."""
        cands = get_candidate_positions(bin_boxes, w, h, d)
        valid = []
        for (x, y, z) in cands:
            if x + w > W or y + h > H or z + d > D:
                continue
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                valid.append((x, y, z))
        if not valid:
            return None
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid[0]

    # ---- Initial packing: Best Fit Decreasing ----
    def initial_packing():
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = [None] * n

        for idx in order:
            w, h, d = boxes[idx]
            best_bin = -1
            best_spot = None
            best_key = None

            for b_idx, bin_boxes in enumerate(bins):
                spot = find_best_spot(bin_boxes, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    # Prefer bin with most volume already (denser packing)
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

        return bins, placements

    # ---- Score computation ----
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
        if container_vol <= 0:
            return float(B)
        return B + 0.1 * (1.0 - last_bin_vol / container_vol)

    # ---- Run initial packing ----
    bins, placements = initial_packing()
    current_score = compute_score(placements)

    # ---- Local search ----
    # Strategy:
    # 1. Try to move boxes from non-last bins into the last bin (reduces B if last bin becomes empty of others, or fills last bin)
    # 2. Try to consolidate: move all boxes from the last bin into other bins to eliminate it
    # 3. Try 2-box swaps to fill the last bin more
    # 4. Try to move boxes between bins to reduce total bin count

    def get_bin_boxes(placements, bin_id):
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]

    def try_move_box(placements, box_idx, target_bin_id):
        """Try to move box_idx to target_bin_id. Returns (success, new_placements) or (False, None)."""
        w, h, d = placements[box_idx][4], placements[box_idx][5], placements[box_idx][6]
        target_boxes = get_bin_boxes(placements, target_bin_id)
        spot = find_best_spot(target_boxes, w, h, d)
        if spot is None:
            return False, None
        new_placements = list(placements)
        new_placements[box_idx] = (target_bin_id, spot[0], spot[1], spot[2], w, h, d)
        return True, new_placements

    def try_consolidate_bin(placements, bin_to_remove):
        """Try to move all boxes from bin_to_remove into other bins. Returns new_placements or None."""
        boxes_in_bin = [i for i in range(n) if placements[i] is not None and placements[i][0] == bin_to_remove]
        if not boxes_in_bin:
            return placements  # Already empty

        # Sort by volume descending for better packing
        boxes_in_bin.sort(key=lambda i: -volumes[i])

        # Build working state
        work_placements = list(placements)
        work_bins = {}
        for b_id in set(p[0] for p in work_placements):
            if b_id != bin_to_remove:
                work_bins[b_id] = get_bin_boxes(work_placements, b_id)

        for box_idx in boxes_in_bin:
            w, h, d = boxes[box_idx]
            placed = False
            # Try each other bin
            for b_id, b_boxes in work_bins.items():
                spot = find_best_spot(b_boxes, w, h, d)
                if spot is not None:
                    work_bins[b_id].append((spot[0], spot[1], spot[2], w, h, d))
                    work_placements[box_idx] = (b_id, spot[0], spot[1], spot[2], w, h, d)
                    placed = True
                    break
            if not placed:
                return None  # Failed, revert

        return work_placements

    improved = True
    max_passes = 200
    pass_count = 0

    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        if time.time() > deadline - 5:
            break

        last_bin_id = max(p[0] for p in placements)
        all_bin_ids = sorted(set(p[0] for p in placements))

        # --- Phase A: Try to move boxes from other bins into the last bin ---
        # This fills the last bin, improving the 0.1*(1-frac) term
        for i in order:
            if time.time() > deadline - 5:
                break
            if placements[i] is None:
                continue
            orig_bin = placements[i][0]
            if orig_bin == last_bin_id:
                continue

            w, h, d = boxes[i]
            target_boxes = get_bin_boxes(placements, last_bin_id)
            spot = find_best_spot(target_boxes, w, h, d)

            if spot is not None:
                old_p = placements[i]
                new_placements = list(placements)
                new_placements[i] = (last_bin_id, spot[0], spot[1], spot[2], w, h, d)
                new_score = compute_score(new_placements)
                if new_score < current_score - 1e-12:
                    placements = new_placements
                    current_score = new_score
                    improved = True
                    last_bin_id = max(p[0] for p in placements)

        # --- Phase B: Try to eliminate the last bin by moving its contents elsewhere ---
        if time.time() > deadline - 5:
            break

        last_bin_id = max(p[0] for p in placements)
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]

        if len(last_bin_boxes) <= 5:  # Only try if last bin is small
            new_placements = try_consolidate_bin(placements, last_bin_id)
            if new_placements is not None:
                new_score = compute_score(new_placements)
                if new_score < current_score - 1e-12:
                    placements = new_placements
                    current_score = new_score
                    improved = True

        # --- Phase C: Try to reduce bin count by moving boxes between non-last bins ---
        if time.time() > deadline - 10:
            break

        all_bin_ids = sorted(set(p[0] for p in placements))
        if len(all_bin_ids) > 1:
            # For each pair of bins, try moving a box from one to the other
            # to potentially empty a bin
            for i in order:
                if time.time() > deadline - 10:
                    break
                if placements[i] is None:
                    continue
                src_bin = placements[i][0]
                w, h, d = boxes[i]

                for tgt_bin in all_bin_ids:
                    if tgt_bin == src_bin:
                        continue
                    tgt_boxes = get_bin_boxes(placements, tgt_bin)
                    spot = find_best_spot(tgt_boxes, w, h, d)
                    if spot is not None:
                        old_p = placements[i]
                        new_placements = list(placements)
                        new_placements[i] = (tgt_bin, spot[0], spot[1], spot[2], w, h, d)
                        new_score = compute_score(new_placements)
                        if new_score < current_score - 1e-12:
                            placements = new_placements
                            current_score = new_score
                            improved = True
                            # Recompute bin ids
                            all_bin_ids = sorted(set(p[0] for p in placements))
                            break  # Break inner loop, this box moved

        # --- Phase D: 2-box swaps to fill last bin ---
        if time.time() > deadline - 15:
            break

        last_bin_id = max(p[0] for p in placements)
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
        other_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] != last_bin_id]

        for i in last_bin_boxes:
            if time.time() > deadline - 15:
                break
            w_i, h_i, d_i = boxes[i]

            for j in other_boxes:
                if time.time() > deadline - 15:
                    break
                w_j, h_j, d_j = boxes[j]
                j_bin = placements[j][0]

                # Remove i from last bin, j from its bin
                last_bin_boxes_without_i = [p[1:7] for p in placements
                                            if p is not None and p[0] == last_bin_id and
                                            not (p[1] == placements[i][1] and p[2] == placements[i][2] and p[3] == placements[i][3])]
                j_bin_boxes_without_j = [p[1:7] for p in placements
                                         if p is not None and p[0] == j_bin and
                                         not (p[1] == placements[j][1] and p[2] == placements[j][2] and p[3] == placements[j][3])]

                spot_j = find_best_spot(last_bin_boxes_without_i, w_j, h_j, d_j)
                if spot_j is None:
                    continue
                spot_i = find_best_spot(j_bin_boxes_without_j, w_i, h_i, d_i)
                if spot_i is None:
                    continue

                old_i = placements[i]
                old_j = placements[j]
                new_placements = list(placements)
                new_placements[i] = (j_bin, spot_i[0], spot_i[1], spot_i[2], w_i, h_i, d_i)
                new_placements[j] = (last_bin_id, spot_j[0], spot_j[1], spot_j[2], w_j, h_j, d_j)

                new_score = compute_score(new_placements)
                if new_score < current_score - 1e-12:
                    placements = new_placements
                    current_score = new_score
                    improved = True
                    # Update lists
                    last_bin_boxes = [i2 for i2 in range(n) if placements[i2] is not None and placements[i2][0] == last_bin_id]
                    other_boxes = [i2 for i2 in range(n) if placements[i2] is not None and placements[i2][0] != last_bin_id]

    # ---- Verification ----
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

    if not is_valid(placements):
        # Fallback: Simple shelf packing to ensure validity
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return placements
