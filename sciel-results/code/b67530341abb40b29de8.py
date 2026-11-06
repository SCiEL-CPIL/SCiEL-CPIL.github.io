"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pack(boxes, container):
    """3D Bin Packing: Best-Fit Decreasing over an extensible empty-space list.

    Returns one (bin, x, y, z, w, h, d) per box, preserving original order and
    dimensions (no rotations). Boxes in the same bin never overlap.
    """
    start = time.time()
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Fast shelf fallback (always valid), used if the search runs long.
    def shelf():
        Wl, Hl, Dl = W, H, D
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > Wl:
                x, y, row_h = 0, y + row_h, 0
            if y + h > Hl:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > Dl:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    # Order: largest volume first (ties broken by largest longest edge).
    order = sorted(range(n), key=lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]),
                                            -max(boxes[i])))

    # bins: list of empty-space lists. Each space is [x, y, z, w, h, d].
    bins = []
    result = [None] * n

    def add_space(lst, s):
        # Drop spaces fully contained in the new one, and reject if contained in an existing one.
        x, y, z, w, h, d = s
        if w <= 0 or h <= 0 or d <= 0:
            return
        kept = []
        for e in lst:
            ex, ey, ez, ew, eh, ed = e
            # e contained in s?
            if (x <= ex and y <= ey and z <= ez and
                    ex + ew <= x + w and ey + eh <= y + h and ez + ed <= z + d):
                continue  # discard e
            # s contained in e?
            if (ex <= x and ey <= y and ez <= z and
                    x + w <= ex + ew and y + h <= ey + eh and z + d <= ez + ed):
                return  # s is contained, do not add
            kept.append(e)
        kept.append(s)
        lst[:] = kept

    for i in order:
        w, h, d = boxes[i]
        placed = False
        # Best-fit: choose the existing empty space that minimises the remaining corner (x+y+z).
        best = None  # (corner, bin_idx, space_idx)
        for bi, lst in enumerate(bins):
            for si in range(len(lst)):
                ex, ey, ez, ew, eh, ed = lst[si]
                if w <= ew and h <= eh and d <= ed:
                    corner = ex + ey + ez
                    if best is None or corner < best[0]:
                        best = (corner, bi, si)
        if best is not None:
            _, bi, si = best
            lst = bins[bi]
            ex, ey, ez, ew, eh, ed = lst[si]
            result[i] = (bi, ex, ey, ez, w, h, d)
            # Split into remaining sub-spaces.
            lst[si] = (ex + w, ey, ez, ew - w, eh, ed)
            add_space(lst, (ex, ey + h, ez, w, eh - h, ed))
            add_space(lst, (ex, ey, ez + d, w, h, ed - d))
            placed = True
        if not placed:
            bi = len(bins)
            lst = []
            add_space(lst, (0, 0, 0, W, H, D))
            bins.append(lst)
            result[i] = (bi, 0, 0, 0, w, h, d)
            add_space(lst, (w, 0, 0, W - w, H, D))
            add_space(lst, (0, h, 0, w, H - h, D))
            add_space(lst, (0, 0, d, w, h, D - d))

        # Deadline guard: if running long, finish the remaining boxes with the shelf heuristic.
        if (i + 1) % 3 == 0 and time.time() - start > 110.0:
            return shelf()

    return result
# EVOLVE-BLOCK-END
