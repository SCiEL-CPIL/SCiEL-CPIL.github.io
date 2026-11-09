"""Rewritten 3D bin packing solver: Space-Pruning Best-Fit Decreasing + aggressive local search."""
import time
import random

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    W, H, D = container
    container_vol = W * H * D

    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # Each bin is a list of (x, y, z, w, h, d)
    bins = []

    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and x + w > bx and y < by + bh and y + h > by and z < bz + bd and z + d > bz:
                return False
        return True

    def candidate_positions(bin_boxes, w, h, d):
        # Generate candidate positions: origin and adjacent to existing boxes
        cands = [(0, 0, 0)]
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Adjacent in +x, +y, +z directions
            cands.append((bx + bw, by, bz))
            cands.append((bx, by + bh, bz))
            cands.append((bx, by, bz + bd))
            # Also try combinations for better gap filling
            for bx2, by2, bz2, bw2, bh2, bd2 in bin_boxes:
                if bx2 != bx or by2 != by or bz2 != bz:
                    cands.append((bx + bw, by2 + bh2, bz))
                    cands.append((bx + bw, by, bz2 + bd2))
                    cands.append((bx2 + bw2, by + bh, bz))
        # Filter valid candidates and remove duplicates
        seen = set()
        unique_cands = []
        for cx, cy, cz in cands:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                unique_cands.append((cx, cy, cz))
        return unique_cands

    def score_position(bin_boxes, x, y, z, w, h, d):
        # Best-fit: minimize the sum of distances to existing boxes (tighter packing)
        if not bin_boxes:
            return 0
        total = 0
        for bx, by, bz, bw, bh, bd in bin_boxes:
            dx = max(0, max(bx - (x + w), x - (bx + bw)))
            dy = max(0, max(by - (y + h), y - (by + bh)))
            dz = max(0, max(bz - (z + d), z - (bz + bd)))
            total += dx * dx + dy * dy + dz * dz
        return total

    def place_box(box_idx, w, h, d):
        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi, bin_boxes in enumerate(bins):
            cands = candidate_positions(bin_boxes, w, h, d)
            for cx, cy, cz in cands:
                sc = score_position(bin_boxes, cx, cy, cz, w, h, d)
                if sc < best_score:
                    best_score = sc
                    best_bin = bi
                    best_pos = (cx, cy, cz)

        if best_bin == -1:
            bins.append([(0, 0, 0, w, h, d)])
            return len(bins) - 1, (0, 0, 0)
        else:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            return best_bin, best_pos

    # Place all boxes
    placements = [None] * n
    for idx, (orig_idx, (w, h, d)) in enumerate(indexed):
        if time.time() > DEADLINE:
            break
        bi, pos = place_box(orig_idx, w, h, d)
        placements[orig_idx] = (bi, pos[0], pos[1], pos[2], w, h, d)

    # Check if all placed; if not, fill remaining with a simple shelf in a new bin
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[i] = (bi, 0, 0, 0, w, h, d)

    # Local improvement: try to move boxes to reduce bin count or improve last-bin fill
    if time.time() < DEADLINE - 10:
        max_bins = max(p[0] for p in placements) if placements else 0
        for _ in range(100):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for i in range(n):
                bi, x, y, z, w, h, d = placements[i]
                # Try to move to a lower bin
                for target_bin in range(bi - 1, -1, -1):
                    # Remove box from current bin
                    bins[bi].remove((x, y, z, w, h, d))
                    cands = candidate_positions(bins[target_bin], w, h, d)
                    found = False
                    for cx, cy, cz in cands:
                        found = True
                        break
                    if found:
                        bins[target_bin].append((cx, cy, cz, w, h, d))
                        placements[i] = (target_bin, cx, cy, cz, w, h, d)
                        improved = True
                        break
                    else:
                        bins[bi].append((x, y, z, w, h, d))
                        break
                if improved:
                    break
            if not improved:
                break

    return placements
