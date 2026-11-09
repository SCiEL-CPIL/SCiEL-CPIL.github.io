"""Improved 3D container loading with efficient skyline-based placement."""
# EVOLVE-BLOCK-START
import time
import itertools

_DEADLINE_MARGIN = 10.0
_HARD_LIMIT = 150.0


def _all_rotations(dims):
    w, h, d = dims
    seen = set()
    rots = []
    for p in itertools.permutations((w, h, d)):
        if p not in seen:
            seen.add(p)
            rots.append(p)
    return rots


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()

    box_rots = [_all_rotations(boxes[i]) for i in range(n)]

    # Sort by volume descending
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Skyline: list of (x, y, z, h, d) representing top surfaces
    # Each entry means: at position (x,y) with height z, there's free space above
    # We use a simpler approach: maintain list of placed boxes and use
    # a "max z" per (x,y) column for quick rejection, then verify.

    # For efficiency, use a dict mapping (x,y) -> max_z_occupied
    max_z = {}  # (x, y) -> highest z that is occupied (exclusive)
    placed_boxes = []  # (x, y, z, w, h, d)

    def can_place(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check against max_z for all (x', y') in the footprint
        for dx in range(w):
            for dy in range(h):
                key = (x + dx, y + dy)
                mz = max_z.get(key, 0)
                if z < mz:
                    return False
        return True

    def mark_placed(x, y, z, w, h, d):
        for dx in range(w):
            for dy in range(h):
                key = (x + dx, y + dy)
                new_z = z + d
                if key not in max_z or new_z > max_z[key]:
                    max_z[key] = new_z
        placed_boxes.append((x, y, z, w, h, d))

    for idx in order:
        if time.time() - start_time > _HARD_LIMIT - _DEADLINE_MARGIN:
            break

        w, h, d = boxes[idx]
        placed = False
        for rot in box_rots[idx]:
            rw, rh, rd = rot
            # Try to find position: scan z, y, x
            found = False
            for z in range(0, D - rd + 1):
                if found:
                    break
                for y in range(0, H - rh + 1):
                    if found:
                        break
                    for x in range(0, W - rw + 1):
                        if can_place(x, y, z, rw, rh, rd):
                            mark_placed(x, y, z, rw, rh, rd)
                            placements[idx] = (0, x, y, z, rw, rh, rd)
                            found = True
                            break
            if found:
                placed = True
                break

    return placements

# EVOLVE-BLOCK-END
