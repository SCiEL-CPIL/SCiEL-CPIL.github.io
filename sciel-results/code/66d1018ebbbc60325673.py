"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import product

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])

    # We'll track placements per bin
    # Each bin has a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists of (x, y, z, w, h, d)
    bin_volumes = []  # occupied volume per bin

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def can_place_in_bin(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if overlaps(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                return False
        return True

    def get_candidate_positions(bin_boxes, w, h, d):
        """Generate candidate positions for a box in a bin."""
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))

        # For each placed box, generate positions adjacent to it
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Right of box
            candidates.add((bx + bw, by, bz))
            # Front of box
            candidates.add((bx, by + bh, bz))
            # Above box
            candidates.add((bx, by, bz + bd))
            # Combinations: right and front
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            # All three
            candidates.add((bx + bw, by + bh, bz + bd))

        # Filter valid positions
        valid = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place_in_bin(bin_boxes, x, y, z, w, h, d):
                    valid.append((x, y, z))
        return valid

    def score_position(x, y, z, w, h, d):
        """Score a position: prefer lower, more compact positions."""
        # Penalize floating (higher z is worse), prefer packing toward origin
        # Use a weighted sum: z has highest weight, then y, then x
        # Also penalize gaps: distance from existing boxes
        return z * 1000 + y * 100 + x

    def best_fit_position(bin_boxes, w, h, d):
        """Find the best position in a bin using best-fit decreasing heuristic."""
        candidates = get_candidate_positions(bin_boxes, w, h, d)
        if not candidates:
            return None
        # Sort by score (lower is better)
        best = min(candidates, key=lambda pos: score_position(pos[0], pos[1], pos[2], w, h, d))
        return best

    def try_place_in_bin(bin_idx, w, h, d):
        """Try to place a box in a specific bin, return position or None."""
        if bin_idx < 0 or bin_idx >= len(bins):
            return None
        pos = best_fit_position(bins[bin_idx], w, h, d)
        if pos is None:
            return None
        x, y, z = pos
        bins[bin_idx].append((x, y, z, w, h, d))
        bin_volumes[bin_idx] += w * h * d
        return (x, y, z)

    def try_place_any_bin(w, h, d):
        """Try to place a box in the best existing bin."""
        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi in range(len(bins)):
            pos = best_fit_position(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                s = score_position(x, y, z, w, h, d)
                if s < best_score:
                    best_score = s
                    best_bin = bi
                    best_pos = pos

        if best_bin == -1:
            return -1, None
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        bin_volumes[best_bin] += w * h * d
        return best_bin, best_pos

    def new_bin():
        """Create a new bin and return its index."""
        bins.append([])
        bin_volumes.append(0)
        return len(bins) - 1

    # Place boxes
    placements_by_box = [None] * n  # (bin_idx, x, y, z, w, h, d)

    for orig_idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline - 5.0:
            break

        # Try existing bins first (best fit)
        bin_idx, pos = try_place_any_bin(w, h, d)

        if bin_idx == -1:
            # Create new bin
            bin_idx = new_bin()
            pos = (0, 0, 0)
            bins[bin_idx].append((0, 0, 0, w, h, d))
            bin_volumes[bin_idx] = w * h * d

        x, y, z = pos
        placements_by_box[orig_idx] = (bin_idx, x, y, z, w, h, d)

    # Check if we have a complete solution
    if any(p is None for p in placements_by_box):
        # Fallback: use shelf heuristic for remaining boxes
        for i in range(n):
            if placements_by_box[i] is not None:
                continue
            w, h, d = boxes[i]
            # Try to place in existing bins
            placed = False
            for bi in range(len(bins)):
                pos = best_fit_position(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    bin_volumes[bi] += w * h * d
                    placements_by_box[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = new_bin()
                bins[bi].append((0, 0, 0, w, h, d))
                bin_volumes[bi] = w * h * d
                placements_by_box[i] = (bi, 0, 0, 0, w, h, d)

    # Build result
    result = []
    for i in range(n):
        p = placements_by_box[i]
        if p is None:
            # Should not happen, but safety
            w, h, d = boxes[i]
            bi = new_bin()
            bins[bi].append((0, 0, 0, w, h, d))
            p = (bi, 0, 0, 0, w, h, d)
        result.append(p)

    return result
# EVOLVE-BLOCK-END
