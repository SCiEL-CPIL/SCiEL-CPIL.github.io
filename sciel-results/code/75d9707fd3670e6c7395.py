import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin well under 120s

    # Permutations of axis rotations: (w,h,d) -> 6 distinct orientations
    def rotations(w, h, d):
        vals = sorted((w, h, d), reverse=True)
        a, b, c = vals
        out = [(a, b, c)]
        for r in [(a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
            if r not in out:
                out.append(r)
        return out

    # Build candidate positions: dense grid over the container
    # Use a grid resolution that's dense enough to find good spots but bounded.
    # We'll place each box at the first (lowest z, then y, then x) grid position
    # where it fits without overlap, trying orientations that minimize the
    # "height" impact (greedy toward flat packing).

    # Grid step: choose based on container size to keep candidate count bounded.
    # We want enough resolution to pack well. Use a step that yields a few hundred
    # positions per axis.
    def grid_step(L):
        # aim for ~40-60 positions per axis
        if L <= 0:
            return 1
        # step so that L/step is between 20 and 80
        target = 50
        step = L / target
        # snap to a "nice" value but keep it >= 1
        if step < 1:
            step = 1
        return step

    sx = grid_step(W)
    sy = grid_step(H)
    sz = grid_step(D)

    # Precompute candidate x, y, z positions (sorted ascending)
    xs = []
    v = 0
    while v <= W - 1e-9:
        xs.append(v)
        v += sx
    ys = []
    v = 0
    while v <= H - 1e-9:
        ys.append(v)
        v += sy
    zs = []
    v = 0
    while v <= D - 1e-9:
        zs.append(v)
        v += sz

    # Overlap test: box at (x,y,z) with dims (w,h,d) overlaps placed boxes?
    # Use a list of placed (x,y,z,w,h,d).
    placed = []  # list of (x,y,z,w,h,d)

    def fits(x, y, z, w, h, d):
        # bounds
        if x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
            return False
        # overlap with placed
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px + 1e-9 or px + pw <= x + 1e-9 or
                    y + h <= py + 1e-9 or py + ph <= y + 1e-9 or
                    z + d <= pz + 1e-9 or pz + pd <= z + 1e-9):
                return False
        return True

    # Sort boxes: place larger boxes first (by volume) to reduce fragmentation.
    # Keep track of original index for output.
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    # For each box (in order), try orientations (prefer flatter ones: smaller max dim on z),
    # and for each orientation scan candidate positions in a "bottom-left-front" order
    # (low z, low y, low x) and place at the first fit.

    result = [None] * n  # each will be (bin, x, y, z, w, h, d)

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        # orientations sorted by "footprint height" (d) then by w*h ascending to prefer flat
        ors = rotations(w, h, d)
        ors.sort(key=lambda o: (o[2], o[0]*o[1]))
        placed_this = False
        for (ow, oh, od) in ors:
            if time.time() > deadline:
                break
            # scan positions: z ascending, y ascending, x ascending
            for z in zs:
                if placed_this:
                    break
                if z + od > D + 1e-9:
                    continue
                for y in ys:
                    if placed_this:
                        break
                    if y + oh > H + 1e-9:
                        continue
                    for x in xs:
                        if x + ow > W + 1e-9:
                            break
                        if fits(x, y, z, ow, oh, od):
                            placed.append((x, y, z, ow, oh, od))
                            result[idx] = (0, x, y, z, ow, oh, od)
                            placed_this = True
                            break
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any unplaced (due to deadline) as left out
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)

    return result
