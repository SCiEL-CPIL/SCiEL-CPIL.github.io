"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Pre-compute box volumes
    box_vols = [w * h * d for (w, h, d) in boxes]

    def evaluate(placements):
        """Compute score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return float('inf')
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return float('inf')
        frac = last_vol / container_vol
        score = B + 0.1 * (1 - frac)
        return score

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def max_cubes_pack(box_list, bin_order_key=None, candidate_sort_key=None, max_candidates=20):
        """
        Best-fit decreasing with candidate position generation.
        box_list: list of (idx, (w, h, d)) sorted in desired packing order.
        """
        # bins[bi] = list of (x, y, z, w, h, d)
        bins = []
        placements_by_idx = {}

        def candidate_positions(bin_boxes, w, h, d):
            """Generate candidate (x,y,z) positions from box corners."""
            cands = [(0, 0, 0)]
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                cands.append((bx + bw, by, bz))
                cands.append((bx, by + bh, bz))
                cands.append((bx, by, bz + bd))
                cands.append((bx + bw, by + bh, bz))
                cands.append((bx + bw, by, bz + bd))
                cands.append((bx, by + bh, bz + bd))
                cands.append((bx + bw, by + bh, bz + bd))

            seen = set()
            valid_cands = []
            for (cx, cy, cz) in cands:
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                key = (cx, cy, cz)
                if key in seen:
                    continue
                seen.add(key)
                valid_cands.append(key)

            if not valid_cands:
                if w <= W and h <= H and d <= D:
                    valid_cands = [(0, 0, 0)]
                else:
                    valid_cands = []

            if candidate_sort_key:
                valid_cands.sort(key=candidate_sort_key)
            else:
                valid_cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))

            return valid_cands[:max_candidates]

        def can_place(bin_boxes, x, y, z, w, h, d):
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if is_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    return False
            return True

        for idx, (w, h, d) in box_list:
            placed = False
            if bins:
                # Order bins by the provided key (default: volume desc)
                if bin_order_key:
                    bins_sorted = sorted(range(len(bins)), key=lambda bi: bin_order_key(bi), reverse=True)
                else:
                    bin_vols = [sum(bw*bh*bd for (bx, by, bz, bw, bh, bd) in bins[bi]) for bi in range(len(bins))]
                    bins_sorted = sorted(range(len(bins)), key=lambda bi: bin_vols[bi], reverse=True)

                for bi in bins_sorted:
                    bb = bins[bi]
                    cands = candidate_positions(bb, w, h, d)
                    for (cx, cy, cz) in cands:
                        if can_place(bb, cx, cy, cz, w, h, d):
                            bb.append((cx, cy, cz, w, h, d))
                            placements_by_idx[idx] = (bi, cx, cy, cz, w, h, d)
                            placed = True
                            break
                    if placed:
                        break

            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements_by_idx[idx] = (bi, 0, 0, 0, w, h, d)

        # Convert to original order
        result = []
        for idx in range(n):
            if idx in placements_by_idx:
                b, x, y, z, pw, ph, pd = placements_by_idx[idx]
                result.append((b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2]))
            else:
                result.append((0, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2]))
        return result

    def local_search(placements, bins_list):
        """
        Try to reduce B by moving boxes from highest-numbered bins to lower ones.
        Returns modified placements and bins_list.
        """
        # Rebuild bins from placements
        bins = {b: [] for b in range(len(bins_list))}
        for p in placements:
            b, x, y, z, w, h, d = p
            bins[b].append((x, y, z, w, h, d))

        def can_place_in_bin(bin_boxes, x, y, z, w, h, d):
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if is_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    return False
            return True

        def get_candidates(bin_boxes, w, h, d):
            cands = [(0, 0, 0)]
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                cands.append((bx + bw, by, bz))
                cands.append((bx, by + bh, bz))
                cands.append((bx, by, bz + bd))
                cands.append((bx + bw, by + bh, bz))
                cands.append((bx + bw, by, bz + bd))
                cands.append((bx, by + bh, bz + bd))
                cands.append((bx + bw, by + bh, bz + bd))
            seen = set()
            valid = []
            for c in cands:
                if c[0] < 0 or c[1] < 0 or c[2] < 0:
                    continue
                if c[0] + w > W or c[1] + h > H or c[2] + d > D:
                    continue
                if c in seen:
                    continue
                seen.add(c)
                valid.append(c)
            valid.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
            return valid[:10]

        # Try to empty bins from highest to lowest
        for _pass in range(2):
            if time.time() > deadline:
                break
            occupied = sorted([b for b in bins if bins[b]], reverse=True)
            for b_id in occupied:
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                boxes_to_try = list(bins[b_id])
                for (x, y, z, w, h, d) in boxes_to_try:
                    # Check if box still in this bin
                    if (x, y, z, w, h, d) not in bins[b_id]:
                        continue
                    moved = False
                    # Try other bins, prefer higher volume
                    other_bins = [b for b in bins if b != b_id and bins[b]]
                    other_bins.sort(key=lambda b: -sum(bw*bh*bd for (bx, by, bz, bw, bh, bd) in bins[b]))
                    for target_b in other_bins:
                        cands = get_candidates(bins[target_b], w, h, d)
                        for (cx, cy, cz) in cands:
                            if can_place_in_bin(bins[target_b], cx, cy, cz, w, h, d):
                                # Remove from source
                                bins[b_id].remove((x, y, z, w, h, d))
                                # Add to target
                                bins[target_b].append((cx, cy, cz, w, h, d))
                                moved = True
                                break
                        if moved:
                            break

        # Rebuild placements
        new_placements = [None] * n
        for b_id, bin_boxes in bins.items():
            for (x, y, z, w, h, d) in bin_boxes:
                # Find original index
                # This is O(n) per box, but n=40 so it's fine
                for idx in range(n):
                    if boxes[idx] == (w, h, d) and new_placements[idx] is None:
                        new_placements[idx] = (b_id, x, y, z, w, h, d)
                        break
                # If not found (duplicate dims), just assign to first available
                if all(new_placements[idx] is not None for idx in range(n)):
                    break

        # Handle any unassigned (shouldn't happen)
        for idx in range(n):
            if new_placements[idx] is None:
                new_placements[idx] = (0, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])

        return new_placements

    # Generate multiple packing strategies
    best_placements = None
    best_score = float('inf')

    def try_packing(box_list, bin_order_key=None, candidate_sort_key=None):
        nonlocal best_placements, best_score
        if time.time() > deadline:
            return
        pl = max_cubes_pack(box_list, bin_order_key, candidate_sort_key)
        s = evaluate(pl)
        if s < best_score:
            best_score = s
            best_placements = pl

    # Strategy 1: Volume descending
    indexed_by_vol = sorted(enumerate(boxes), key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
    try_packing(indexed_by_vol)

    # Strategy 2: Max dimension descending
    indexed_by_maxdim = sorted(enumerate(boxes), key=lambda t: -max(t[1]))
    try_packing(indexed_by_maxdim)

    # Strategy 3: Sum of dimensions descending
    indexed_by_sum = sorted(enumerate(boxes), key=lambda t: -(t[1][0] + t[1][1] + t[1][2]))
    try_packing(indexed_by_sum)

    # Strategy 4: Randomized restarts
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        # Randomly perturb the volume order
        indexed = list(enumerate(boxes))
        random.shuffle(indexed)
        # Sort by volume but with noise
        indexed.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2] + random.uniform(-0.5, 0.5) * max(box_vols)))
        try_packing(indexed)

    # Local search on best placement
    if best_placements is not None and time.time() < deadline:
        # Extract bin count
        bins_used = set(p[0] for p in best_placements)
        bins_list = list(range(max(bins_used) + 1))
        improved = local_search(best_placements, bins_list)
        s_improved = evaluate(improved)
        if s_improved < best_score:
            best_score = s_improved
            best_placements = improved

    if best_placements is None:
        # Fallback
        best_placements = [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return list(best_placements)
# EVOLVE-BLOCK-END
