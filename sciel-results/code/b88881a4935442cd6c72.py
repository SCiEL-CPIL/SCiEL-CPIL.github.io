import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    # Sort indices by various heuristics
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    min_dim_sorted = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted]
    for _ in range(4):
        random_order = list(range(n))
        random.shuffle(random_order)
        orderings.append(random_order)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def validate_placement(placements):
        if len(placements) != n:
            return False
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            bins[b].append((x, y, z, w, h, d))
        for bin_idx, box_list in bins.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    x1, y1, z1, w1, h1, d1 = box_list[i]
                    x2, y2, z2, w2, h2, d2 = box_list[j]
                    if (x1 < x2 + w2 and x2 < x1 + w1 and
                        y1 < y2 + h2 and y2 < y1 + h1 and
                        z1 < z2 + d2 and z2 < z1 + d1):
                        return False
        return True

    def find_best_position(w, h, d, bin_boxes):
        """Find the best position for a box in a bin using corner-based candidates."""
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions from existing box edges
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])

        for bx, by, bz, bw, bh, bd in bin_boxes:
            x_cands.add(bx)
            x_cands.add(bx + bw)
            y_cands.add(by)
            y_cands.add(by + bh)
            z_cands.add(bz)
            z_cands.add(bz + bd)

        # Filter valid candidates
        x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
        y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
        z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

        if not x_cands or not y_cands or not z_cands:
            return None

        best_pos = None
        best_cost = float('inf')

        # Bottom-left-far heuristic: minimize z, then y, then x
        # Weight z most heavily to fill from bottom
        for z in sorted(z_cands):
            for y in sorted(y_cands):
                for x in sorted(x_cands):
                    # Check overlap with all boxes in bin
                    overlaps = False
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if (x < bx + bw and bx < x + w and
                            y < by + bh and by < y + h and
                            z < bz + bd and bz < z + d):
                            overlaps = True
                            break
                    if overlaps:
                        continue

                    # Cost: prefer positions that are "tucked in" - low z, low y, low x
                    cost = z * (H + 1) * (W + 1) + y * (W + 1) + x

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        if best_pos == (0, 0, 0):
                            return best_pos

        return best_pos

    def greedy_place(ordering, bin_selection='fullest'):
        """Greedy placement with configurable bin selection strategy."""
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)
        bin_volumes = []  # track volume used in each bin

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                if bin_selection == 'fullest':
                    # Try bins with most volume used first
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
                else:
                    # Try bins with least volume used first (balance)
                    bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi])
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = find_best_position(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_improve(placements):
        """Try to improve placement by moving boxes to fill bins more tightly."""
        if time.time() > deadline:
            return placements

        # Group boxes by bin
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bins[b].append((i, x, y, z, w, h, d))

        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1

            for bin_idx in list(bins.keys()):
                if time.time() > deadline:
                    break

                bin_boxes = bins[bin_idx]
                for i, (box_i, x, y, z, w, h, d) in enumerate(bin_boxes):
                    # Try to remove this box and re-place it in another bin
                    remaining = [b for j, b in enumerate(bin_boxes) if j != i]

                    # Try to place in other bins
                    best_new_bin = None
                    best_new_pos = None
                    best_cost = float('inf')

                    for other_bin in bins:
                        if other_bin == bin_idx:
                            continue
                        other_boxes = [(x2, y2, z2, w2, h2, d2) for (_, x2, y2, z2, w2, h2, d2) in bins[other_bin]]
                        pos = find_best_position(w, h, d, other_boxes)
                        if pos is not None:
                            # Cost: prefer moving to bins that are already being used
                            # and positions that are more compact
                            cost = z * 100 + y * 10 + x
                            if cost < best_cost:
                                best_cost = cost
                                best_new_bin = other_bin
                                best_new_pos = pos

                    if best_new_bin is not None:
                        # Remove from current bin
                        bins[bin_idx].pop(i)
                        if not bins[bin_idx]:
                            del bins[bin_idx]

                        # Add to new bin
                        nx, ny, nz = best_new_pos
                        bins[best_new_bin].append((box_i, nx, ny, nz, w, h, d))

                        # Update placement
                        placements[box_i] = (best_new_bin, nx, ny, nz, w, h, d)
                        improved = True

        return placements

    best_result = None
    best_score = float('inf')

    # Try different strategies
    strategies = [
        ('greedy_fullest', lambda order: greedy_place(order, 'fullest')),
        ('greedy_balance', lambda order: greedy_place(order, 'balance')),
    ]

    for strategy_name, strategy_func in strategies:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = strategy_func(order)

                if not validate_placement(placements):
                    continue

                # Try local improvement
                if time.time() < deadline - 20:
                    improved_placements = local_improve(placements)
                    if validate_placement(improved_placements):
                        placements = improved_placements

                score = compute_score(placements)

                if score < best_score:
                    best_score = score
                    best_result = placements[:]
            except Exception:
                continue

    if best_result is not None:
        return best_result

    # Fallback: one box per bin
    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
