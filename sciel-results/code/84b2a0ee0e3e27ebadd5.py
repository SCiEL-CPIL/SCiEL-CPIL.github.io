"""3D Container Loading: Multi-strategy with spatial index and local improvement."""
import time
import math
from itertools import product

# EVOLVE-BLOCK-START

def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin vs 120s limit

    W, H, D = container
    n = len(boxes)
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_vol = W * H * D

    # Precompute orientations for each box
    box_orients = []
    box_volumes = []
    for w, h, d in boxes:
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                if perm[0] <= W and perm[1] <= H and perm[2] <= D:
                    orients.append(perm)
        if not orients:
            orients = [(w, h, d)]
        box_orients.append(orients)
        box_volumes.append(w * h * d)

    def try_pack(order, max_candidates=200):
        """Try to pack boxes in given order using best-fit decreasing."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * n

        # Spatial index: list of placed boxes for overlap checking
        def check_overlap(cx, cy, cz, bw, bh, bd):
            for (px, py, pz, pw, ph, pd) in placed:
                if not ((cx + bw <= px) or (px + pw <= cx) or
                        (cy + bh <= py) or (py + ph <= cy) or
                        (cz + bd <= pz) or (pz + pd <= cz)):
                    return True
            return False

        def get_candidate_positions():
            """Generate candidate positions from placed boxes."""
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            return sorted(cx_set), sorted(cy_set), sorted(cz_set)

        for idx in order:
            if time.time() > deadline:
                break

            orients = box_orients[idx]
            best_pos = None
            best_orient = None
            best_score = None

            cx_list, cy_list, cz_list = get_candidate_positions()

            # Limit candidate positions for speed
            all_candidates = []
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        all_candidates.append((cx, cy, cz))
            all_candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            all_candidates = all_candidates[:max_candidates]

            for (cx, cy, cz) in all_candidates:
                for (bw, bh, bd) in orients:
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    if check_overlap(cx, cy, cz, bw, bh, bd):
                        continue
                    score = (cz, cy, cx, max(bw, bh, bd))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (bw, bh, bd)

            if best_pos is not None:
                cx, cy, cz = best_pos
                bw, bh, bd = best_orient
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
            else:
                w, h, d = boxes[idx]
                results[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill any remaining None
        for i in range(n):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        return results, placed

    def compute_score(results):
        total_vol = 0
        for r in results:
            if r[0] == 0:
                total_vol += r[4] * r[5] * r[6]
        return total_vol / container_vol if container_vol > 0 else 0

    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    results1, placed1 = try_pack(order1)
    score1 = compute_score(results1)

    # Strategy 2: Max dimension descending
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    results2, placed2 = try_pack(order2)
    score2 = compute_score(results2)

    # Strategy 3: Volume ascending (small first to fill gaps)
    order3 = sorted(range(n), key=lambda i: box_volumes[i])
    results3, placed3 = try_pack(order3)
    score3 = compute_score(results3)

    # Strategy 4: Mixed - large first, then small
    large = [i for i in range(n) if box_volumes[i] >= median_vol] if n > 0 else []
    small = [i for i in range(n) if box_volumes[i] < median_vol] if n > 0 else []
    median_vol = sorted(box_volumes)[n // 2] if n > 0 else 0
    order4 = large + small
    results4, placed4 = try_pack(order4)
    score4 = compute_score(results4)

    # Pick best strategy
    best_score = max(score1, score2, score3, score4)
    if best_score == score1:
        best_results, best_placed = results1, placed1
    elif best_score == score2:
        best_results, best_placed = results2, placed2
    elif best_score == score3:
        best_results, best_placed = results3, placed3
    else:
        best_results, best_placed = results4, placed4

    # Phase 2: Local improvement - try to place unplaced boxes in gaps
    if time.time() < deadline - 5.0:
        unplaced = [i for i in range(n) if best_results[i][0] == -1]
        if unplaced:
            # Sort by volume ascending
            unplaced.sort(key=lambda i: box_volumes[i])
            placed = list(best_placed)
            results = list(best_results)

            def check_overlap2(cx, cy, cz, bw, bh, bd, placed_list):
                for (px, py, pz, pw, ph, pd) in placed_list:
                    if not ((cx + bw <= px) or (px + pw <= cx) or
                            (cy + bh <= py) or (py + ph <= cy) or
                            (cz + bd <= pz) or (pz + pd <= cz)):
                        return True
                return False

            for idx in unplaced:
                if time.time() > deadline:
                    break
                orients = box_orients[idx]
                best_pos = None
                best_orient = None
                best_score = None

                cx_set = {0}
                cy_set = {0}
                cz_set = {0}
                for (px, py, pz, pw, ph, pd) in placed:
                    cx_set.add(px)
                    cx_set.add(px + pw)
                    cy_set.add(py)
                    cy_set.add(py + ph)
                    cz_set.add(pz)
                    cz_set.add(pz + pd)

                all_candidates = []
                for cz in sorted(cz_set):
                    for cy in sorted(cy_set):
                        for cx in sorted(cx_set):
                            all_candidates.append((cx, cy, cz))
                all_candidates.sort(key=lambda c: (c[2], c[1], c[0]))
                all_candidates = all_candidates[:150]

                for (cx, cy, cz) in all_candidates:
                    for (bw, bh, bd) in orients:
                        if cx + bw > W or cy + bh > H or cz + bd > D:
                            continue
                        if check_overlap2(cx, cy, cz, bw, bh, bd, placed):
                            continue
                        score = (cz, cy, cx, max(bw, bh, bd))
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (bw, bh, bd)

                if best_pos is not None:
                    cx, cy, cz = best_pos
                    bw, bh, bd = best_orient
                    placed.append((cx, cy, cz, bw, bh, bd))
                    results[idx] = (0, cx, cy, cz, bw, bh, bd)

            # Update best if improved
            new_score = compute_score(results)
            if new_score > best_score:
                best_results = results
                best_score = new_score

    return best_results

# EVOLVE-BLOCK-END
