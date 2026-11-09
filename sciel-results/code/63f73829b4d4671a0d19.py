"""Rewritten loader: best-fit packing with rotations, skyline-style placement + grid-accelerated overlap + multi-strategy restarts."""
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

    def run_packing(order, score_fn, deadline):
        """Run one greedy pass with given box order and position scoring.
        Returns (placements_list, filled_volume)."""
        # Placed boxes: list of (x, y, z, w, h, d)
        placed = []
        # Spatial grid for fast overlap: cell size = min dim of container / small factor
        # Use a dict mapping (cx, cy, cz) -> list of placed box indices
        # Cell size: use a reasonable value so each cell has few boxes
        cell_size = max(1, min(W, H, D) // 8)
        grid = {}

        def add_to_grid(x, y, z, w, h, d, idx):
            x0 = x // cell_size
            x1 = (x + w - 1) // cell_size
            y0 = y // cell_size
            y1 = (y + h - 1) // cell_size
            z0 = z // cell_size
            z1 = (z + d - 1) // cell_size
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    for cz in range(z0, z1 + 1):
                        grid.setdefault((cx, cy, cz), []).append(idx)

        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check only boxes in overlapping grid cells
            x0 = x // cell_size
            x1 = (x + w - 1) // cell_size
            y0 = y // cell_size
            y1 = (y + h - 1) // cell_size
            z0 = z // cell_size
            z1 = (z + d - 1) // cell_size
            checked = set()
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    for cz in range(z0, z1 + 1):
                        for pi in grid.get((cx, cy, cz), ()):
                            if pi in checked:
                                continue
                            checked.add(pi)
                            px, py, pz, pw, ph, pd = placed[pi]
                            if not (x + w <= px or px + pw <= x or
                                    y + h <= py or py + ph <= y or
                                    z + d <= pz or pz + pd <= z):
                                return False
            return True

        def candidate_positions():
            cands = set()
            cands.add((0, 0, 0))
            for px, py, pz, pw, ph, pd in placed:
                cands.add((px + pw, py, pz))   # right
                cands.add((px, py + ph, pz))   # on top
                cands.add((px, py, pz + pd))   # front
            return list(cands)

        placements = [None] * n
        filled_vol = 0

        for idx in order:
            if time.time() > deadline:
                break
            w, h, d = boxes[idx]
            rots = rotations(w, h, d)
            best = None
            best_score = None

            cands = candidate_positions()
            # Sort candidates by the scoring function
            cands.sort(key=lambda c: score_fn(c))

            for rw, rh, rd in rots:
                found = False
                for (cx, cy, cz) in cands:
                    if can_place(cx, cy, cz, rw, rh, rd):
                        s = score_fn((cx, cy, cz))
                        if best_score is None or s < best_score:
                            best_score = s
                            best = (cx, cy, cz, rw, rh, rd)
                        found = True
                        break  # first valid for this rotation (lowest by sort)
                if found:
                    break

            if best is not None:
                cx, cy, cz, rw, rh, rd = best
                placed.append((cx, cy, cz, rw, rh, rd))
                add_to_grid(cx, cy, cz, rw, rh, rd, len(placed) - 1)
                placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                filled_vol += rw * rh * rd
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, filled_vol

    # Define several scoring functions (lower is better)
    def score_low(c):
        return (c[2], c[1], c[0])

    def score_corner(c):
        # Prefer positions close to origin in all axes, weighted
        return (c[0] + c[1] + c[2], c[2], c[1], c[0])

    def score_shelf(c):
        # Prefer lower z, then tighter x,y
        return (c[2], c[0], c[1])

    # Define several box orderings
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    order_vol_desc = sorted(range(n), key=lambda i: -volumes[i])
    order_vol_asc = sorted(range(n), key=lambda i: volumes[i])
    # Sort by max dimension descending
    max_dims = [max(boxes[i]) for i in range(n)]
    order_max_dim = sorted(range(n), key=lambda i: -max_dims[i])
    # Sort by min dimension descending
    min_dims = [min(boxes[i]) for i in range(n)]
    order_min_dim = sorted(range(n), key=lambda i: -min_dims[i])
    # Random order
    random.seed(42)
    order_rand = list(range(n))
    random.shuffle(order_rand)

    # Multi-strategy restarts: try different (order, score) combos
    strategies = [
        (order_vol_desc, score_low),
        (order_vol_desc, score_corner),
        (order_vol_desc, score_shelf),
        (order_max_dim, score_low),
        (order_max_dim, score_corner),
        (order_min_dim, score_low),
        (order_vol_asc, score_low),
        (order_rand, score_low),
        (order_rand, score_corner),
    ]

    best_placements = None
    best_vol = -1

    for si, (order, sfn) in enumerate(strategies):
        if time.time() > deadline:
            break
        # Give each strategy a sub-deadline so we can try several
        sub_deadline = min(deadline, start + 140.0 - (len(strategies) - si) * 0.5)
        # Actually, just use the main deadline but break early if too slow
        placements, vol = run_packing(order, sfn, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
        # If we already filled a large fraction, we can stop early
        if container_vol > 0 and best_vol / container_vol > 0.85:
            break

    if best_placements is None:
        # Fallback: single pass with volume-desc + low score
        best_placements, _ = run_packing(order_vol_desc, score_low, deadline)

    return best_placements
# EVOLVE-BLOCK-END
