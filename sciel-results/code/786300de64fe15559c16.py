"""Rewritten loader: best-fit packing with rotations, skyline-style placement + time guard."""
import time
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

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

    # Sort boxes by volume descending to place large ones first
    indexed = sorted(range(len(boxes)), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    # Placed boxes: list of (x, y, z, w, h, d)
    placed = []

    def can_place(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def candidate_positions(w, h, d):
        # Generate candidate (x,y,z) positions:
        # 1. Origin
        # 2. On top of each placed box
        # 3. To the right of each placed box
        # 4. In front of each placed box
        cands = set()
        cands.add((0, 0, 0))
        for px, py, pz, pw, ph, pd in placed:
            cands.add((px + pw, py, pz))   # right
            cands.add((px, py + ph, pz))   # on top
            cands.add((px, py, pz + pd))   # front
        # Also try grid-aligned positions near existing boxes
        return list(cands)

    placements = [None] * len(boxes)

    for idx in indexed:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = rotations(w, h, d)
        best = None
        best_key = None

        for rw, rh, rd in rots:
            # Try to find a valid position
            # Start with simple heuristic: scan along axes
            # Use a more systematic approach: try all candidate positions
            cands = candidate_positions(rw, rh, rd)
            # Sort candidates by (z, y, x) to prefer lower positions
            cands.sort(key=lambda c: (c[2], c[1], c[0]))

            for (cx, cy, cz) in cands:
                if can_place(cx, cy, cz, rw, rh, rd):
                    # Prefer positions that are "tight" (close to origin)
                    key = (cz, cy, cx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (cx, cy, cz, rw, rh, rd)
                    break  # first valid position for this rotation is enough (lowest)

            # If we found a placement for this rotation, consider it
            if best is not None:
                break

        if best is not None:
            cx, cy, cz, rw, rh, rd = best
            placed.append((cx, cy, cz, rw, rh, rd))
            placements[idx] = (0, cx, cy, cz, rw, rh, rd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any None slots (shouldn't happen, but safe)
    for i in range(len(placements)):
        if placements[i] is None:
            placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return placements
# EVOLVE-BLOCK-END
