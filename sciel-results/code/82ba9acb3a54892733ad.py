"""Rewritten loader: best-fit packing with rotations, skyline-style placement + time guard."""
import time
import math
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)
    container_vol = W * H * D

    # Precompute all 6 rotations per box
    def rotations(w, h, d):
        seen = set()
        out = []
        for a, b, c in ((w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)):
            key = (a, b, c)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def try_place_all(order, placed, placements, deadline):
        """Try to place all boxes in given order. Returns (filled_vol, placed, placements)."""
        placed = []
        placements = [None] * n

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in placed:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True

        def candidate_positions():
            cands = set()
            cands.add((0, 0, 0))
            for px, py, pz, pw, ph, pd in placed:
                cands.add((px + pw, py, pz))
                cands.add((px, py + ph, pz))
                cands.add((px, py, pz + pd))
                # Also try corners: combinations of face positions
                cands.add((px + pw, py + ph, pz))
                cands.add((px + pw, py, pz + pd))
                cands.add((px, py + ph, pz + pd))
            return cands

        filled_vol = 0

        for idx in order:
            if time.time() > deadline:
                break
            w, h, d = boxes[idx]
            rots = rotations(w, h, d)
            best = None
            best_key = None

            cands = candidate_positions()
            cands_sorted = sorted(cands, key=lambda c: (c[2], c[1], c[0]))

            for rw, rh, rd in rots:
                for (cx, cy, cz) in cands_sorted:
                    if can_place(cx, cy, cz, rw, rh, rd):
                        key = (cz, cy, cx)
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (cx, cy, cz, rw, rh, rd)
                        break  # first valid for this rotation

                if best is not None:
                    break

            if best is not None:
                cx, cy, cz, rw, rh, rd = best
                placed.append((cx, cy, cz, rw, rh, rd))
                placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                filled_vol += rw * rh * rd
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill any None slots
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return filled_vol, placed, placements

    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    # Strategy 2: Max dimension descending (tall/wide boxes first)
    order2 = sorted(range(n), key=lambda i: -max(boxes[i]))

    # Strategy 3: Volume / max_dim (flatter boxes first)
    order3 = sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(boxes[i]))

    # Strategy 4: Volume ascending (small first, fill gaps)
    order4 = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])

    best_vol = -1
    best_placements = None
    best_placed = None

    # Try each strategy
    strategies = [order1, order2, order3, order4]
    for s_idx, order in enumerate(strategies):
        if time.time() > deadline:
            break
        vol, placed, placements = try_place_all(order, None, None, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
            best_placed = placed

    # Random restarts with perturbations on the best-performing base order
    if time.time() < deadline - 10:
        base_order = order1  # volume descending as base
        n_boxes = n
        # Number of restarts depends on remaining time
        elapsed = time.time() - start
        remaining = deadline - time.time()
        # Estimate time per restart from first strategy run
        per_restart = max(0.1, elapsed / max(1, len(strategies)))
        max_restarts = int(min(remaining / per_restart, 200))

        for r in range(max_restarts):
            if time.time() > deadline:
                break

            # Create perturbed order: swap random pairs
            order = list(base_order)
            n_swaps = max(1, n_boxes // 4)
            for _ in range(n_swaps):
                i, j = random.randint(0, n_boxes - 1), random.randint(0, n_boxes - 1)
                order[i], order[j] = order[j], order[i]

            vol, placed, placements = try_place_all(order, None, None, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
                best_placed = placed

    # Fill any None slots in best result
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    else:
        for i in range(n):
            if best_placements[i] is None:
                best_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return best_placements
# EVOLVE-BLOCK-END
