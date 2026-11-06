"""Model-and-optimize 3D bin packing with deadline guard."""
import time
import random
import math
from typing import List, Tuple
from collections import defaultdict


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 115.0

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D

    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)

    def find_position(existing_items, w, h, d, W, H, D):
        """
        Find a position (x, y, z) where box (w, h, d) fits without overlap.
        Uses all unique coordinate combinations from box faces.
        """
        if not existing_items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None

        # Collect unique x, y, z coordinates from box faces
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
            z_coords.add(ez)
            z_coords.add(ez + ed)

        # Filter to only coordinates where the box would fit
        x_coords = sorted(x for x in x_coords if x + w <= W)
        y_coords = sorted(y for y in y_coords if y + h <= H)
        z_coords = sorted(z for z in z_coords if z + d <= D)

        if not x_coords or not y_coords or not z_coords:
            return None

        # Generate candidate positions and sort by (z, y, x) - bottom-left-front
        candidates = []
        for cx in x_coords:
            for cy in y_coords:
                for cz in z_coords:
                    candidates.append((cx, cy, cz))

        candidates.sort(key=lambda c: (c[2], c[1], c[0]))

        for (cx, cy, cz) in candidates:
            # Check overlap with all existing items
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                return (cx, cy, cz)

        return None

    def find_all_positions(existing_items, w, h, d, W, H, D, max_candidates=50):
        """
        Find multiple valid positions for a box, sorted by compactness (cx+cy+cz).
        Returns a list of positions.
        """
        if not existing_items:
            if w <= W and h <= H and d <= D:
                return [(0, 0, 0)]
            return []

        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
            z_coords.add(ez)
            z_coords.add(ez + ed)

        x_coords = sorted(x for x in x_coords if x + w <= W)
        y_coords = sorted(y for y in y_coords if y + h <= H)
        z_coords = sorted(z for z in z_coords if z + d <= D)

        if not x_coords or not y_coords or not z_coords:
            return []

        candidates = []
        for cx in x_coords:
            for cy in y_coords:
                for cz in z_coords:
                    candidates.append((cx, cy, cz))

        candidates.sort(key=lambda c: (c[2], c[1], c[0]))

        valid_positions = []
        for (cx, cy, cz) in candidates:
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                valid_positions.append((cx, cy, cz))
                if len(valid_positions) >= max_candidates:
                    break

        return valid_positions

    def bin_fragmentation(items, W, H, D):
        """
        Compute a fragmentation score for a bin: higher is worse.
        """
        if not items:
            return 0.0
        max_x = max(x + w for x, y, z, w, h, d in items)
        max_y = max(y + h for x, y, z, w, h, d in items)
        max_z = max(z + d for x, y, z, w, h, d in items)
        bbox_vol = max_x * max_y * max_z
        filled_vol = sum(w * h * d for x, y, z, w, h, d in items)
        if bbox_vol == 0:
            return 0.0
        return 1.0 - filled_vol / bbox_vol

    def bin_score(state, W, H, D):
        """
        Score for a bin: lower is better for placing a new box.
        Prefers bins with higher fill fraction and lower fragmentation.
        """
        if not state:
            return float('inf')
        filled = sum(w * h * d for _, _, _, w, h, d in state)
        fill_frac = filled / container_vol
        frag = bin_fragmentation(state, W, H, D)
        # Heavier penalty for fragmentation, prefer well-filled bins
        return -2.0 * fill_frac + 1.0 * frag

    def pack_with_order(ordering):
        """Pack boxes in given order. Returns placements list or None."""
        state = {}
        placements = [None] * n
        all_placed = True

        for box_idx in ordering:
            if time.time() > deadline:
                all_placed = False
                break

            w, h, d = boxes[box_idx]

            # Try existing bins: prefer bin with best score
            best_bin = None
            best_pos = None
            best_score_val = float('inf')

            bin_ids = list(state.keys())
            bin_scores = {bid: bin_score(state[bid], W, H, D) for bid in bin_ids}
            bin_ids_sorted = sorted(bin_ids, key=lambda bid: bin_scores[bid])

            for bin_id in bin_ids_sorted:
                pos = find_position(state[bin_id], w, h, d, W, H, D)
                if pos is not None:
                    # Prefer positions that minimize coordinate sum (compactness)
                    waste = pos[0] + pos[1] + pos[2]
                    # Also consider how well the box fits in this bin's remaining space
                    bin_items = state[bin_id]
                    max_x = max((x + bw for x, y, z, bw, bh, bd in bin_items), default=0)
                    max_y = max((y + bh for x, y, z, bw, bh, bd in bin_items), default=0)
                    max_z = max((z + bd for x, y, z, bw, bh, bd in bin_items), default=0)
                    # Reward placing in bins where the box extends the bounding box less
                    extension_penalty = max(0, pos[0] + w - max_x) + max(0, pos[1] + h - max_y) + max(0, pos[2] + d - max_z)
                    total_score = waste + 0.3 * extension_penalty
                    if total_score < best_score_val:
                        best_score_val = total_score
                        best_bin = bin_id
                        best_pos = pos

            if best_bin is not None:
                state[best_bin] = state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                new_bin_id = max(state.keys()) + 1 if state else 0
                state[new_bin_id] = [(0, 0, 0, w, h, d)]
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)

        if all_placed and all(p is not None for p in placements):
            return placements
        return None

    def local_search_polish(placements, W, H, D, max_iterations=300):
        """
        Local search: try to move boxes to other bins, reposition within bins,
        or swap boxes to reduce total bins used or improve last-bin fill.
        """
        if not placements:
            return placements

        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)

        current_score = compute_score(placements, W, H, D)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1

            # Try repositioning each box within its current bin to more compact positions
            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                
                other_items = [(placements[j][1], placements[j][2], placements[j][3],
                               placements[j][4], placements[j][5], placements[j][6])
                              for j in bin_boxes[b] if j != box_idx]
                
                # Find all valid positions and pick the most compact one
                positions = find_all_positions(other_items, w, h, d, W, H, D, max_candidates=30)
                if positions:
                    best_pos = min(positions, key=lambda p: p[0] + p[1] + p[2])
                    if best_pos != (x, y, z):
                        new_placements = list(placements)
                        new_placements[box_idx] = (b, best_pos[0], best_pos[1], best_pos[2], w, h, d)
                        new_score = compute_score(new_placements, W, H, D)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            improved = True

            # Try moving each box to another bin if it fits
            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]

                other_bins = [bid for bid in bin_boxes if bid != b]
                # Sort target bins by how well they'd accept this box
                scored_targets = []
                for target_bin in other_bins:
                    target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_boxes[target_bin]]
                    pos = find_position(target_items, w, h, d, W, H, D)
                    if pos is not None:
                        scored_targets.append((pos[0] + pos[1] + pos[2], target_bin, pos))
                
                scored_targets.sort(key=lambda x: x[0])
                for _, target_bin, pos in scored_targets:
                    new_placements = list(placements)
                    new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                    new_score = compute_score(new_placements, W, H, D)
                    if new_score < current_score:
                        placements = new_placements
                        current_score = new_score
                        bin_boxes[b].remove(box_idx)
                        if not bin_boxes[b]:
                            del bin_boxes[b]
                        bin_boxes[target_bin].append(box_idx)
                        improved = True
                        break

            # Try swapping pairs of boxes in different bins
            for box_idx1 in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b1, x1, y1, z1, w1, h1, d1 = placements[box_idx1]
                for box_idx2 in range(box_idx1 + 1, n):
                    if time.time() > deadline - 5:
                        break
                    b2, x2, y2, z2, w2, h2, d2 = placements[box_idx2]
                    if b1 == b2:
                        continue

                    bin2_items = [(placements[j][1], placements[j][2], placements[j][3],
                                   placements[j][4], placements[j][5], placements[j][6])
                                  for j in bin_boxes[b2] if j != box_idx2]
                    pos1_in_b2 = find_position(bin2_items, w1, h1, d1, W, H, D)
                    if pos1_in_b2 is None:
                        continue

                    bin1_items = [(placements[j][1], placements[j][2], placements[j][3],
                                   placements[j][4], placements[j][5], placements[j][6])
                                  for j in bin_boxes[b1] if j != box_idx1]
                    pos2_in_b1 = find_position(bin1_items, w2, h2, d2, W, H, D)
                    if pos2_in_b1 is None:
                        continue

                    new_placements = list(placements)
                    new_placements[box_idx1] = (b2, pos1_in_b2[0], pos1_in_b2[1], pos1_in_b2[2], w1, h1, d1)
                    new_placements[box_idx2] = (b1, pos2_in_b1[0], pos2_in_b1[1], pos2_in_b1[2], w2, h2, d2)
                    new_score = compute_score(new_placements, W, H, D)
                    if new_score < current_score:
                        placements = new_placements
                        current_score = new_score
                        bin_boxes[b1].remove(box_idx1)
                        bin_boxes[b2].append(box_idx1)
                        bin_boxes[b2].remove(box_idx2)
                        bin_boxes[b1].append(box_idx2)
                        improved = True
                        break

        return placements

    def final_polish(placements, W, H, D):
        """
        High-precision polish: iteratively reposition boxes to minimize
        the sum of coordinates (compactness) within each bin, and try
        to consolidate bins.
        """
        if not placements:
            return placements

        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)

        current_score = compute_score(placements, W, H, D)
        improved = True
        iterations = 0
        max_iters = 80

        while improved and iterations < max_iters and time.time() < deadline - 2:
            improved = False
            iterations += 1

            # For each bin, try to reposition all boxes to minimize total coordinate sum
            for bin_id in list(bin_boxes.keys()):
                if time.time() > deadline - 2:
                    break
                box_indices = bin_boxes[bin_id]
                if len(box_indices) <= 1:
                    continue

                # Greedy reposition: sort by volume desc, reposition each
                sorted_indices = sorted(box_indices, key=lambda i: box_vols[i], reverse=True)
                
                for box_idx in sorted_indices:
                    b, x, y, z, w, h, d = placements[box_idx]
                    
                    other_items = [(placements[j][1], placements[j][2], placements[j][3],
                                   placements[j][4], placements[j][5], placements[j][6])
                                  for j in box_indices if j != box_idx]
                    
                    positions = find_all_positions(other_items, w, h, d, W, H, D, max_candidates=50)
                    if positions:
                        # Pick position with minimum cx+cy+cz
                        best_pos = min(positions, key=lambda p: p[0] + p[1] + p[2])
                        if best_pos != (x, y, z):
                            new_placements = list(placements)
                            new_placements[box_idx] = (b, best_pos[0], best_pos[1], best_pos[2], w, h, d)
                            new_score = compute_score(new_placements, W, H, D)
                            if new_score < current_score:
                                placements = new_placements
                                current_score = new_score
                                improved = True

            # Try to consolidate: move boxes from last bin to earlier bins
            if len(bin_boxes) > 1:
                bins_sorted = sorted(bin_boxes.keys())
                last_bin = bins_sorted[-1]
                if last_bin in bin_boxes:
                    for box_idx in list(bin_boxes[last_bin]):
                        if time.time() > deadline - 2:
                            break
                        b, x, y, z, w, h, d = placements[box_idx]
                        # Try all earlier bins, prefer ones with most space
                        for target_bin in bins_sorted[:-1]:
                            target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                             placements[j][4], placements[j][5], placements[j][6])
                                            for j in bin_boxes[target_bin]]
                            pos = find_position(target_items, w, h, d, W, H, D)
                            if pos is not None:
                                new_placements = list(placements)
                                new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                                new_score = compute_score(new_placements, W, H, D)
                                if new_score < current_score:
                                    placements = new_placements
                                    current_score = new_score
                                    bin_boxes[b].remove(box_idx)
                                    if not bin_boxes[b]:
                                        del bin_boxes[b]
                                    bin_boxes[target_bin].append(box_idx)
                                    improved = True
                                    break

        return placements

    # Main optimization: try multiple orderings
    best_result = None
    best_score = float('inf')

    orderings = []

    # 1. Volume descending
    orderings.append(indices_sorted)

    # 2. Volume ascending
    orderings.append(list(reversed(indices_sorted)))

    # 3. Original order
    orderings.append(indices)

    # 4. Max dimension descending
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))

    # 5. Surface area descending
    orderings.append(sorted(indices, key=lambda i: 2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]), reverse=True))

    # 6. Aspect ratio based
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1], boxes[i][2]) / max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))

    # 7. Randomized volume descending
    for seed in [42, 123]:
        random.seed(seed)
        shuffled = indices_sorted[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    # 8. Weighted by volume^0.5
    orderings.append(sorted(indices, key=lambda i: math.sqrt(box_vols[i]), reverse=True))

    # 9. Sort by minimum dimension
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))

    # 10. Sort by maximum dimension
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))

    # 11. Sort by (w+h+d) descending
    orderings.append(sorted(indices, key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True))

    # 12. Sort by min(w,h) descending
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1]), reverse=True))

    # Run all orderings and keep the best
    for ordering in orderings:
        if time.time() > deadline - 20:
            break

        result = pack_with_order(ordering)
        if result is not None:
            score = compute_score(result, W, H, D)
            if score < best_score:
                best_score = score
                best_result = result

    # If no valid result found, fall back to simple shelf
    if best_result is None:
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
        best_result = placements

    # Local search polish to improve packing
    if best_result and time.time() < deadline - 20:
        best_result = local_search_polish(best_result, W, H, D)
        best_score = compute_score(best_result, W, H, D)

    # Final high-precision polish
    if best_result and time.time() < deadline - 5:
        best_result = final_polish(best_result, W, H, D)
        best_score = compute_score(best_result, W, H, D)

    # Post-process: relabel bins to be consecutive starting from 0
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]

    return best_result


def compute_score(placements, W, H, D):
    if not placements:
        return 0.0

    bins_used = set(p[0] for p in placements)
    B = len(bins_used)

    last_bin = max(bins_used)

    container_vol = W * H * D
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)

    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
