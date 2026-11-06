"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _get_rotations(w, h, d):
    """Return all unique axis rotations of a box (w, h, d)."""
    rots = [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]
    seen = set()
    out = []
    for r in rots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def _find_2d_position(rw, rd, W, D, rects):
    """
    Find a 2D position (x, z) for a rect of size (rw, rd) in a W x D plane,
    given a list of occupied rects [(x0, x1, z0, z1), ...].
    Uses bottom-left search: scan candidate x positions (0 and right edges),
    then candidate z positions (0 and top edges of x-overlapping rects).
    Returns (x, z) or None.
    """
    if rw > W or rd > D:
        return None
    # Candidate x positions
    x_candidates = [0]
    for (x0, x1, z0, z1) in rects:
        if x1 not in x_candidates:
            x_candidates.append(x1)
    x_candidates = [x for x in x_candidates if x + rw <= W]
    x_candidates.sort()

    for x in x_candidates:
        # Candidate z positions: 0 and top edges of rects that overlap in x
        z_candidates = [0]
        for (x0, x1, z0, z1) in rects:
            if x0 < x + rw and x < x1:  # overlap in x
                if z1 not in z_candidates:
                    z_candidates.append(z1)
        z_candidates = [z for z in z_candidates if z + rd <= D]
        z_candidates.sort()

        for z in z_candidates:
            # Check overlap
            ok = True
            for (x0, x1, z0, z1) in rects:
                if not (x + rw <= x0 or x >= x1 or z + rd <= z0 or z >= z1):
                    ok = False
                    break
            if ok:
                return (x, z)
    return None

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 135.0  # 150s limit, 15s safety margin

    # Precompute rotations for each box
    box_rots = [_get_rotations(w, h, d) for (w, h, d) in boxes]

    def try_pack(order):
        """
        order: list of box indices in the order to place.
        Uses horizontal shelf packing: shelves stacked in H (vertical) direction.
        Within a shelf, 2D bottom-left search in W x D plane.
        Box rotation (rw, rh, rd): rw along x, rh along y (vertical), rd along z (depth).
        Returns (placements, used_vol).
        """
        placements = [None] * n
        used_vol = 0
        shelves = []  # [y_start, y_height, [(x0, x1, z0, z1), ...]]
        y_cursor = 0

        for idx in order:
            rots = box_rots[idx]
            best = None  # (y_start, x, z, rw, rh, rd, shelf_index)

            # Try existing shelves
            for si, shelf in enumerate(shelves):
                y_start, y_h, rects = shelf
                for (rw, rh, rd) in rots:
                    if rh > H - y_start:
                        continue
                    pos = _find_2d_position(rw, rd, W, D, rects)
                    if pos is not None:
                        x, z = pos
                        cand = (y_start, x, z, rw, rh, rd, si)
                        if best is None:
                            best = cand
                        else:
                            if (cand[0], cand[1], cand[2]) < (best[0], best[1], best[2]):
                                best = cand
                        break  # found position in this shelf for this rotation
                if best is not None:
                    break  # found a shelf that works

            if best is None:
                # Try new shelf
                for (rw, rh, rd) in rots:
                    if rh > H - y_cursor:
                        continue
                    if rw <= W and rd <= D:
                        best = (y_cursor, 0, 0, rw, rh, rd, -1)
                        break

            if best is None:
                continue

            y_start, x, z, rw, rh, rd, si = best
            placements[idx] = (x, y_start, z, rw, rh, rd)
            used_vol += rw * rh * rd

            if si == -1:
                shelves.append([y_cursor, rh, [(x, x + rw, z, z + rd)]])
                y_cursor += rh
            else:
                shelves[si][1] = max(shelves[si][1], rh)
                shelves[si][2].append((x, x + rw, z, z + rd))

        return placements, used_vol

    # Generate candidate orderings
    def make_orders():
        orders = []
        vol_order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
        orders.append(vol_order)
        maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
        orders.append(maxdim_order)
        orders.append(list(range(n)))
        rng = random.Random(42)
        for _ in range(200):
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            orders.append(shuffled)
        return orders

    best_placements = [None] * n
    best_vol = 0

    orders = make_orders()
    for order in orders:
        if time.time() > deadline:
            break
        placements, vol = try_pack(order)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements

    result = []
    for i in range(n):
        if best_placements[i] is not None:
            x, y, z, rw, rh, rd = best_placements[i]
            result.append((0, x, y, z, rw, rh, rd))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))

    return result
# EVOLVE-BLOCK-END
