"""3D Container Loading - improved greedy with multi-start and rotation support."""
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin before 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    if n == 0:
        return []

    # Generate all 6 unique rotations for a box (w, h, d)
    def get_rotations(w, h, d):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        return list(rots)

    # Precompute rotations for each box
    box_rots = []
    for w, h, d in boxes:
        box_rots.append(get_rotations(w, h, d))

    # Sort orders to try
    def sort_by_volume(b):
        w, h, d = b
        return -(w * h * d)

    def sort_by_max_dim(b):
        return -max(b)

    def sort_by_aspect(b):
        w, h, d = b
        return -(max(b) / min(b) if min(b) > 0 else 0)

    def sort_by_volume_density(b):
        w, h, d = b
        return -(w * h * d) / (w + h + d + 1e-9)

    sort_keys = [sort_by_volume, sort_by_max_dim, sort_by_aspect, sort_by_volume_density]

    best_placements = None
    best_volume = 0.0
    container_vol = W * H * D

    # Try multiple sorting strategies
    for sort_fn in sort_keys:
        if time.time() > deadline:
            break

        # Create sorted index
        indexed = list(range(n))
        indexed.sort(key=lambda i: sort_fn(boxes[i]))

        # Greedy placement with rotations
        placed = [False] * n
        placements = [None] * n
        placed_boxes = []  # list of (x, y, z, w, h, d) for already placed boxes

        def can_place(x, y, z, bw, bh, bd):
            """Check if a box at (x,y,z) with dims (bw,bh,bd) fits without overlap."""
            if x < 0 or y < 0 or z < 0:
                return False
            if x + bw > W or y + bh > H or z + bd > D:
                return False
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Check overlap
                if (x < px + pw and x + bw > px and
                    y < py + ph and y + bh > py and
                    z < pz + pd and z + bd > pz):
                    return False
            return True

        def find_position(bw, bh, bd):
            """Find a valid position for a box with given dimensions."""
            # Try positions in a heuristic order: prefer lower y, then lower z, then lower x
            # Use a grid-based search with some randomization
            best_pos = None
            best_score = float('inf')

            # Candidate x positions: 0 and right edges of existing boxes
            xs = [0]
            ys = [0]
            zs = [0]
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if px + pw <= W:
                    xs.append(px + pw)
                if py + ph <= H:
                    ys.append(py + ph)
                if pz + pd <= D:
                    zs.append(pz + pd)

            xs = sorted(set(x for x in xs if x <= W - bw))
            ys = sorted(set(y for y in ys if y <= H - bh))
            zs = sorted(set(z for z in zs if z <= D - bd))

            for z in zs:
                for y in ys:
                    for x in xs:
                        if can_place(x, y, z, bw, bh, bd):
                            # Score: prefer lower z, then lower y, then lower x (compact packing)
                            score = z * 10000 + y * 100 + x
                            if score < best_score:
                                best_score = score
                                best_pos = (x, y, z)
                            break  # Only check first valid x for each (y,z)
                    else:
                        continue
                    break  # Only check first valid y for each z
                else:
                    continue
                break

            if best_pos is not None:
                return best_pos

            # Fallback: brute force search on a coarser grid
            step_x = max(1, bw // 10)
            step_y = max(1, bh // 10)
            step_z = max(1, bd // 10)
            for z in range(0, D - bd + 1, step_z):
                for y in range(0, H - bh + 1, step_y):
                    for x in range(0, W - bw + 1, step_x):
                        if can_place(x, y, z, bw, bh, bd):
                            return (x, y, z)
            return None

        total_volume = 0.0
        for idx in indexed:
            if time.time() > deadline:
                break
            w, h, d = boxes[idx]
            placed = False
            for rot in box_rots[idx]:
                bw, bh, bd = rot
                pos = find_position(bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    placements[idx] = (0, x, y, z, bw, bh, bd)
                    placed_boxes.append((x, y, z, bw, bh, bd))
                    total_volume += bw * bh * bd
                    placed = True
                    break
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        if all(p is not None for p in placements):
            if total_volume > best_volume:
                best_volume = total_volume
                best_placements = list(placements)

    # If we found a valid placement, return it
    if best_placements is not None:
        return best_placements

    # Fallback: simple shelf packing (original algorithm)
    placements = []
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + d > D:
            placements.append((-1, 0, 0, 0, w, h, d))
            continue
        placements.append((0, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return placements
