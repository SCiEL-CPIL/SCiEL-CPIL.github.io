import time
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort indices by volume descending, then by max dimension for tie-breaking
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    max_dims = [max(boxes[i]) for i in range(n)]
    sorted_idx = sorted(range(n), key=lambda i: (-volumes[i], -max_dims[i]))

    def try_place(box_idx, bin_placements, W, H, D):
        """Try to place a box in a bin using shelf-based heuristic.
        Returns (x, y, z) if placed, or None."""
        w, h, d = boxes[box_idx]
        if w > W or h > H or d > D:
            return None

        # Find the best shelf position
        # Shelves are defined by existing boxes' y-coordinates
        # Try placing at existing shelf y-levels and at y=0
        shelf_ys = set([0])
        for (bx, by, bz, bw, bh, bd) in bin_placements:
            shelf_ys.add(by)

        best_pos = None
        best_score = float('inf')

        for sy in sorted(shelf_ys):
            if sy + h > H:
                continue
            # Find the shelf: boxes with same y
            shelf_boxes = [(bx, bz, bw, bd) for (bx, by, bz, bw, bh, bd) in bin_placements if by == sy]

            # Sort shelf boxes by x
            shelf_boxes.sort(key=lambda t: t[0])

            # Try placing after each box in the shelf, and at x=0
            candidate_xs = [0]
            for (sx, sz, sw, sd) in shelf_boxes:
                candidate_xs.append(sx + sw)

            for cx in candidate_xs:
                if cx + w > W:
                    continue
                # Check for overlap with all boxes in the bin
                overlap = False
                for (bx, by, bz, bw, bh, bd) in bin_placements:
                    if (cx < bx + bw and bx < cx + w and
                        sy < by + bh and by < sy + h and
                        0 < bz + bd and bz < d):  # z overlap
                        # More precise z check
                        if not (d <= bz or bz + bd <= 0):
                            # Actually need proper 3D overlap check
                            if (cx < bx + bw and bx < cx + w and
                                sy < by + bh and by < sy + h and
                                0 < bz + bd and bz < d):
                                # z ranges: [0, d] vs [bz, bz+bd]
                                if not (d <= bz or bz + bd <= 0):
                                    overlap = True
                                    break
                if not overlap:
                    # Score: prefer lower z, then lower y, then lower x
                    score = (0, sy, cx)
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, sy, 0)

        return best_pos

    def greedy_pack(order):
        """Given an order of box indices, greedily pack them into bins."""
        bins = []  # list of lists of (x, y, z, w, h, d)
        placements = []

        for box_idx in order:
            w, h, d = boxes[box_idx]
            placed = False

            for b in range(len(bins)):
                pos = try_place(box_idx, bins[b], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[b].append((x, y, z, w, h, d))
                    placements.append((b, x, y, z, w, h, d))
                    placed = True
                    break

            if not placed:
                # Try new bin
                if w <= W and h <= H and d <= D:
                    new_b = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    placements.append((new_b, 0, 0, 0, w, h, d))
                else:
                    # Box doesn't fit in any bin - place in a new bin anyway (shouldn't happen)
                    new_b = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    placements.append((new_b, 0, 0, 0, w, h, d))

        return placements

    def score_placements(placements):
        """Compute the score: B + 0.1*(1 - last_bin_volume_fraction)"""
        if not placements:
            return 0.0

        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(bin_labels)

        # Compute volume of last bin
        last_bin_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_bin_vol += p[4] * p[5] * p[6]

        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Initial greedy pack with sorted order
    best_order = sorted_idx[:]
    best_placements = greedy_pack(best_order)
    best_score = score_placements(best_placements)

    # Local search: swap pairs in the order
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_placements

    random.seed(42)
    current_order = best_order[:]
    current_score = best_score

    # Simulated annealing style
    T = 0.1
    T_min = 0.001
    step = 0

    while time.time() < deadline - 5:
        step += 1
        # Pick two random positions to swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue

        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]

        new_placements = greedy_pack(new_order)
        new_score = score_placements(new_placements)

        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current_order = new_order
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_placements = new_placements

        # Cool down
        T = max(T * 0.999, T_min)

        # Check time
        if time.time() > deadline - 5:
            break

    return best_placements
