"""3D Container Loading - fast greedy with rotation support."""
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 105.0  # Leave 15s margin for safety

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

            # Check candidates in order of preference (lowest z, then y, then x)
            # We sort candidates to ensure we pick the most compact position
            candidates = []
            for z in zs:
                for y in ys:
                    for x in xs:
                        # Basic bounds check is already done in xs/ys/zs generation,
                        # but double check for safety against floating point/integer issues if any
                        if 0 <= x and 0 <= y and 0 <= z and x+bw <= W and y+bh <= H and z+bd <= D:
                            candidates.append((z, y, x))
            
            # Sort by z, then y, then x to find the "bottom-left-farthest" (or similar compact) spot
            # Sorting ensures we check the most desirable positions first
            candidates.sort()
            
            for z, y, x in candidates:
                if can_place(x, y, z, bw, bh, bd):
                    return (x, y, z)
            
            # If no candidate position worked, the box likely doesn't fit in the remaining space
            # We do NOT do a brute force fallback as it is too slow.
            return None

        total_volume = 0.0
        for idx in indexed:
            if time.time() > deadline:
                # If we run out of time, we must stop and return what we have so far
                # or break out to handle the fallback
                break
            
            w, h, d = boxes[idx]
            placed = False
            
            # Try rotations
            # To save time, we can limit the number of rotations tried if time is tight
            # But usually 6 rotations is fast enough.
            for rot in box_rots[idx]:
                if time.time() > deadline:
                    break
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

        # If we finished the loop without hitting the deadline, update best
        if time.time() <= deadline and all(p is not None for p in placements):
            if total_volume > best_volume:
                best_volume = total_volume
                best_placements = list(placements)
        else:
            # If we hit deadline or some placements are None (which shouldn't happen if we break cleanly,
            # but if we break mid-box, placements for that box is None),
            # we should probably discard this partial result or mark remaining as -1.
            # The logic above breaks out of the for loop.
            # If we broke out, `placements` might contain None for the current idx and subsequent ones.
            # We need to ensure we return a valid list.
            # Let's just discard this candidate if it's incomplete.
            pass

    # If we found a valid placement, return it
    if best_placements is not None:
        return best_placements

    # Fallback: simple shelf packing (original algorithm)
    # This is very fast and guarantees a result.
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
