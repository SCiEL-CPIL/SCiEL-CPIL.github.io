import time

def pack(boxes, container):
    W, H, D = container
    boxes = list(boxes)
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return []
    start = time.time()
    deadline = start + 114.0  # safety margin vs 120s

    def box_key(w, h, d):
        return (w * h * d, max(w, h, d), w)

    order = sorted(range(n), key=lambda i: box_key(*boxes[i]), reverse=True)

    def find_best(frees, w, h, d):
        best = None
        for (fx, fy, fz, fw, fh, fd) in frees:
            if w > fw or h > fh or d > fd:
                continue
            bx = fx + (fw - w) // 2
            by = fy + (fh - h) // 2
            bz = fz + (fd - d) // 2
            resid = (fw - w) * (fh - h) * (fd - d)
            depth = bx + by + bz
            key = (resid, depth, bx, by, bz)
            if best is None or key < best[0]:
                best = (key, (bx, by, bz))
        return best[1] if best else None

    def split(frees, bx, by, bz, w, h, d):
        new = []
        for (fx, fy, fz, fw, fh, fd) in frees:
            ex = fx + fw
            ey = fy + fh
            ez = fz + fd
            if ex <= bx or ey <= by or ez <= bz or fx >= bx + w or fy >= by + h or fz >= bz + d:
                new.append((fx, fy, fz, fw, fh, fd))
                continue
            if fx < bx:
                new.append((fx, fy, fz, bx - fx, fh, fd))
            if ex > bx + w:
                new.append((bx + w, fy, fz, ex - (bx + w), fh, fd))
            if fy < by:
                new.append((fx, fy, fz, fw, by - fy, fd))
            if ey > by + h:
                new.append((fx, by + h, fz, fw, ey - (by + h), fd))
            if fz < bz:
                new.append((fx, fy, fz, fw, fh, bz - fz))
            if ez > bz + d:
                new.append((fx, fy, bz + d, fw, fh, ez - (bz + d)))
        return new

    def place_all():
        bins = []
        pos = [None] * n
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            for bi in range(len(bins)):
                p = find_best(bins[bi], w, h, d)
                if p is not None:
                    bx, by, bz = p
                    bins[bi] = split(bins[bi], bx, by, bz, w, h, d)
                    pos[idx] = (bi, bx, by, bz)
                    placed = True
                    break
            if not placed:
                p = find_best([(0, 0, 0, W, H, D)], w, h, d)
                if p is None:
                    return None, None
                bx, by, bz = p
                bins.append(split([(0, 0, 0, W, H, D)], bx, by, bz, w, h, d))
                pos[idx] = (len(bins) - 1, bx, by, bz)
        return bins, pos

    bins, pos = place_all()
    if pos is None:
        # Fallback: shelf (guaranteed if boxes individually fit)
        placements = []
        b = x = y = z = row_h = layer_d = 0
        for (w, h, d) in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    def make_placements(pos):
        pl = []
        for i in range(n):
            bi, bx, by, bz = pos[i]
            w, h, d = boxes[i]
            pl.append((bi, bx, by, bz, w, h, d))
        return pl

    placements = make_placements(pos)

    # Local improvement: pull boxes toward origin to tighten last bin.
    def overlaps_at(bi, i, bx, by, bz):
        w, h, d = boxes[i]
        for j in range(n):
            if j == i:
                continue
            p = pos[j]
            if p is None or p[0] != bi:
                continue
            jx, jy, jz = p[1], p[2], p[3]
            jw, jh, jd = boxes[j]
            if bx < jx + jw and bx + w > jx and by < jy + jh and by + h > jy and bz < jz + jd and bz + d > jz:
                return True
        return False

    def try_improve(pos):
        improved = True
        while improved and (time.time() < deadline - 2.0):
            improved = False
            for i in range(n):
                bi, bx, by, bz = pos[i]
                w, h, d = boxes[i]
                best = (bx, by, bz)
                nx = 0
                if not overlaps_at(bi, i, nx, by, bz):
                    cand = (nx, by, bz)
                    if (cand[0] + cand[1] + cand[2]) < (bx + by + bz):
                        best = cand
                ny = 0
                if not overlaps_at(bi, i, best[0], ny, best[2]):
                    cand = (best[0], ny, best[2])
                    if (cand[0] + cand[1] + cand[2]) < (best[0] + best[1] + best[2]):
                        best = cand
                nz = 0
                if not overlaps_at(bi, i, best[0], best[1], nz):
                    cand = (best[0], best[1], nz)
                    if (cand[0] + cand[1] + cand[2]) < (best[0] + best[1] + best[2]):
                        best = cand
                if best != (bx, by, bz):
                    pos[i] = (bi, best[0], best[1], best[2])
                    improved = True
        return pos

    if time.time() < deadline - 2.0:
        pos = try_improve(pos)
        placements = make_placements(pos)

    return placements
