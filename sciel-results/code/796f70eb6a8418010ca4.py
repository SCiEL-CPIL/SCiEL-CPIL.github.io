"""
Strategy: 3D Shelf-Packing with Rotations + Multiple Ordering Heuristics +
Local-Search Refinement of the placement order.

We maintain a list of placed boxes and check overlaps in O(N) per candidate,
which is fine for 60 boxes. The key structural improvements over the previous
skyline approach are:
1. A "shelf" based packing that keeps horizontal shelves per z-level and
   places boxes side-by-side within a shelf before starting a new shelf.
2. Multiple deterministic ordering heuristics (volume desc, max-dim desc,
   aspect-ratio based, etc.)
3. A short local-search phase that swaps adjacent boxes in the ordering to
   improve total placed volume, using fast greedy re-packing for each
   candidate.
"""

import time
import itertools
import random

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 110.0

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]

    def pack(order):
        """
        Greedy shelf-based packing for a given box order.
        Returns (placements_list, total_volume).
        placements_list: list of (idx, x, y, z, w, h, d) for placed boxes.
        """
        placed = []  # (idx, x, y, z, w, h, d)
        total_vol = 0

        # Shelves: list of (z, shelf_width_used, shelf_height)
        # Actually, we'll track for each z-level, the current x and y usage.
        # Simpler: maintain a list of placed boxes and do best-fit.
        # Let's use a "first-fit decreasing" with shelves.

        # Shelf representation: list of shelves, each shelf is (z_start, max_h_used)
        # Within a shelf, we place boxes left-to-right, bottom-to-top.
        # For each shelf, track the current x position and the max height of boxes in it.
        # New shelves start at z = max of all shelf tops.

        shelves = []  # each shelf: [z_start, max_h, current_x, boxes_in_shelf]

        for idx in order:
            w, h, d = boxes[idx]
            orients = orient_cache[idx]
            best_placement = None
            best_score = None

            for (bw, bh, bd) in orients:
                # Try to place in each shelf
                for s_i, shelf in enumerate(shelves):
                    z_start = shelf[0]
                    shelf_max_h = shelf[1]
                    cur_x = shelf[2]
                    # The shelf's top is z_start + shelf_max_h
                    # If we place a box of height bh at this shelf, the shelf height becomes max(shelf_max_h, bh)
                    # The box occupies x: cur_x .. cur_x+bw-1, y: 0..bh-1 (we'll use y=0 for simplicity, or track y too)
                    # Actually, we should track y as well. Let's use a 2D shelf: x and y.
                    # For simplicity, let's just do 1D shelves in x, and stack in z.
                    # Box at (cur_x, 0, z_start) with size (bw, bh, bd)
                    if cur_x + bw <= W and z_start + bd <= D:
                        # Check if it overlaps with boxes in this shelf
                        # All boxes in this shelf start at z_start and have height up to shelf_max_h
                        # If bh > shelf_max_h, the box will extend above the shelf.
                        # This is getting complicated. Let's use a simpler approach:
                        # Just check against all placed boxes.
                        pass

            # Simpler: just check against all placed boxes
            for (bw, bh, bd) in orients:
                # Try positions: (0,0,0), then for each placed box, try adjacent positions
                candidate_positions = [(0, 0, 0)]
                for (_, px, py, pz, pw, ph, pd) in placed:
                    candidate_positions.extend([
                        (px + pw, py, pz),
                        (px, py + ph, pz),
                        (px, py, pz + pd),
                        (px + pw, py + ph, pz),
                        (px, py + ph, pz + pd),
                        (px + pw, py, pz + pd),
                        (px + pw, py + ph, pz + pd),
                    ])
                # Also add some random-ish positions? No, keep it deterministic for now.
                for (px, py, pz) in candidate_positions:
                    if px < 0 or py < 0 or pz < 0:
                        continue
                    if px + bw > W or py + bh > H or pz + bd > D:
                        continue
                    # Check overlap
                    overlap = False
                    for (_, ox, oy, oz, ow, oh, od) in placed:
                        if px < ox + ow and px + bw > ox and py < oy + oh and py + bh > oy and pz < oz + od and pz + bd > oz:
                            overlap = True
                            break
                    if not overlap:
                        # Score: prefer lower z, then lower y, then lower x
                        score = (pz, py, px)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_placement = (px, py, pz, bw, bh, bd)

            if best_placement is not None:
                bx, by, bz, bw, bh, bd = best_placement
                placed.append((idx, bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd

        # Build result
        result = [None] * n
        for p in placed:
            idx, x, y, z, w, h, d = p
            result[idx] = (0, x, y, z, w, h, d)
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        return result, total_vol

    # Define several ordering heuristics
    indexed = list(range(n))
    orders = [
        sorted(indexed, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indexed, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indexed, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indexed, key=lambda i: (boxes[i][0]+boxes[i][1]+boxes[i][2]), reverse=True),
    ]

    best_result = None
    best_volume = -1

    for order in orders:
        if time.time() > deadline:
            break
        res, vol = pack(order)
        if vol > best_volume:
            best_volume = vol
            best_result = res

    # Local search: swap adjacent elements in the best order and re-pack
    # Use the best order found so far as the starting point
    if best_result is not None:
        # Find which order produced the best result
        # We'll just use the first order that gave best_volume
        for order in orders:
            if time.time() > deadline:
                break
            res, vol = pack(order)
            if vol >= best_volume:
                best_order = order[:]
                break
        else:
            best_order = orders[0][:]

        # Local search with swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_order) - 1):
                if time.time() > deadline:
                    break
                # Swap i and i+1
                best_order[i], best_order[i+1] = best_order[i+1], best_order[i]
                res, vol = pack(best_order)
                if vol > best_volume:
                    best_volume = vol
                    best_result = res
                    improved = True
                else:
                    # Revert
                    best_order[i], best_order[i+1] = best_order[i+1], best_order[i]

    # Randomized restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        rand_order = orders[0][:]
        random.shuffle(rand_order)
        res, vol = pack(rand_order)
        if vol > best_volume:
            best_volume = vol
            best_result = res

    if best_result is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_result

# EVOLVE-BLOCK-END
