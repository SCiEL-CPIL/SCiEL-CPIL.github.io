import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    class Bin3D:
        """A bin with a 3D grid occupancy map for fast overlap checks."""
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            # Occupancy grid: 1 if cell is occupied
            self.occ = np.zeros((W, H, D), dtype=np.int8)
            self.boxes = []  # list of (x, y, z, w, h, d)
            self.volume = 0

        def mark_occupied(self, x, y, z, w, h, d):
            self.occ[x:x+w, y:y+h, z:z+d] = 1
            self.boxes.append((x, y, z, w, h, d))
            self.volume += w * h * d

        def is_free(self, x, y, z, w, h, d):
            if x + w > self.W or y + h > self.H or z + d > self.D:
                return False
            return not np.any(self.occ[x:x+w, y:y+h, z:z+d])

        def find_positions(self, w, h, d):
            """Find all feasible positions for a box of size (w,h,d)."""
            if w > self.W or h > self.H or d > self.D:
                return []

            # Generate candidate coordinates from box edges
            x_cands = set([0])
            y_cands = set([0])
            z_cands = set([0])

            for bx, by, bz, bw, bh, bd in self.boxes:
                x_cands.add(bx)
                x_cands.add(bx + bw)
                y_cands.add(by)
                y_cands.add(by + bh)
                z_cands.add(bz)
                z_cands.add(bz + bd)

            # Filter valid candidates
            x_cands = [x for x in x_cands if 0 <= x and x + w <= self.W]
            y_cands = [y for y in y_cands if 0 <= y and y + h <= self.H]
            z_cands = [z for z in z_cands if 0 <= z and z + d <= self.D]

            if not x_cands or not y_cands or not z_cands:
                return []

            positions = []
            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        if self.is_free(x, y, z, w, h, d):
                            positions.append((x, y, z))
            return positions

    def greedy_place(ordering, cost_func):
        """Greedy placement with a given cost function for position selection."""
        placements = [None] * n
        bins = []
        bin_volumes = []

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                # Try bins with most volume used first (fullest first)
                bin_indices = sorted(range(len(bins)), key=lambda bi: bins[bi].volume, reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                positions = bins[bin_i].find_positions(w, h, d)
                if positions:
                    # Choose the best position based on cost function
                    best_pos = None
                    best_cost = float('inf')
                    for (x, y, z) in positions:
                        cost = cost_func(x, y, z, w, h, d, bins[bin_i])
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                    if best_pos is not None:
                        x, y, z = best_pos
                        bins[bin_i].mark_occupied(x, y, z, w, h, d)
                        placements[box_idx] = (bin_i, x, y, z, w, h, d)
                        placed = True
                        break

            if not placed:
                new_bin_idx = len(bins)
                new_bin = Bin3D(W, H, D)
                new_bin.mark_occupied(0, 0, 0, w, h, d)
                bins.append(new_bin)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    # Cost functions
    def cost_bottom_left_near(x, y, z, w, h, d, bin_obj):
        """Prefer positions close to the bottom-left-front corner."""
        return z * (H + 1) * (W + 1) + y * (W + 1) + x

    def cost_best_fit(x, y, z, w, h, d, bin_obj):
        """Prefer positions that minimize the sum of distances to existing boxes (tight packing)."""
        if not bin_obj.boxes:
            return 0
        min_dist = float('inf')
        for bx, by, bz, bw, bh, bd in bin_obj.boxes:
            # Compute minimum distance between the new box and existing box
            dx = max(0, max(x - (bx + bw), bx - (x + w)))
            dy = max(0, max(y - (by + bh), by - (y + h)))
            dz = max(0, max(z - (bz + bd), bz - (z + d)))
            dist = dx + dy + dz
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def cost_combined(x, y, z, w, h, d, bin_obj):
        """Combined cost: prefer low z, then tight packing."""
        base_cost = z * 1000 + y * 10 + x
        if not bin_obj.boxes:
            return base_cost
        min_dist = float('inf')
        for bx, by, bz, bw, bh, bd in bin_obj.boxes:
            dx = max(0, max(x - (bx + bw), bx - (x + w)))
            dy = max(0, max(y - (by + bh), by - (y + h)))
            dz = max(0, max(z - (bz + bd), bz - (z + d)))
            dist = dx + dy + dz
            if dist < min_dist:
                min_dist = dist
        return base_cost + min_dist * 0.1

    def cost_maximize_contact(x, y, z, w, h, d, bin_obj):
        """Prefer positions that maximize contact with existing boxes (minimize gaps)."""
        if not bin_obj.boxes:
            return 0
        max_contact = 0
        for bx, by, bz, bw, bh, bd in bin_obj.boxes:
            # Check if faces are touching
            contact = 0
            # X faces
            if x == bx + bw or bx == x + w:
                # Check Y and Z overlap
                y_overlap = min(y + h, by + bh) - max(y, by)
                z_overlap = min(z + d, bz + bd) - max(z, bz)
                if y_overlap > 0 and z_overlap > 0:
                    contact += y_overlap * z_overlap
            # Y faces
            if y == by + bh or by == y + h:
                x_overlap = min(x + w, bx + bw) - max(x, bx)
                z_overlap = min(z + d, bz + bd) - max(z, bz)
                if x_overlap > 0 and z_overlap > 0:
                    contact += x_overlap * z_overlap
            # Z faces
            if z == bz + bd or bz == z + d:
                x_overlap = min(x + w, bx + bw) - max(x, bx)
                y_overlap = min(y + h, by + bh) - max(y, by)
                if x_overlap > 0 and y_overlap > 0:
                    contact += x_overlap * y_overlap
            if contact > max_contact:
                max_contact = contact
        return -max_contact  # Minimize negative contact = maximize contact

    best_result = None
    best_score = float('inf')

    cost_functions = [cost_bottom_left_near, cost_best_fit, cost_combined, cost_maximize_contact]

    for cost_func in cost_functions:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = greedy_place(order, cost_func)

                if not validate_placement(placements):
                    continue

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
