import time

def pack(boxes, container):
    start = time.time()
    DEADLINE = start + 115.0

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort by volume descending
    indexed = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # Each bin is a heightmap: heights[x][y] = max z at (x,y). W x H.
    bins = []  # list of 2D lists

    def new_heightmap():
        return [[0] * H for _ in range(W)]

    def find_best_pos(heights, w, h, d):
        """Return (x, y, z) for best placement or None."""
        best = None
        best_key = None
        max_x = W - w
        max_y = H - h
        for x in range(max_x + 1):
            col_slice = heights[x]  # not helpful, need 2D
            for x_i in range(x, x + w):
                # quick check col
                pass
            # full check
            for y in range(max_y + 1):
                max_z = 0
                ok = True
                for i in range(x, x + w):
                    row = heights[i]
                    for j in range(y, y + h):
                        z = row[j]
                        if z + d > D:
                            ok = False
                            break
                        if z > max_z:
                            max_z = z
                    if not ok:
                        break
                if ok:
                    key = (max_z, x, y)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (x, y, max_z)
        return best

    def stamp(heights, x, y, z, w, h, d):
        for i in range(x, x + w):
            row = heights[i]
            for j in range(y, y + h):
                row[j] = z + d

    placements = []
    placed = [False] * n

    for idx in indexed:
        if time.time() > DEADLINE:
            break
        w, h, d = boxes[idx]
        placed_flag = False

        for b_idx in range(len(bins)):
            if time.time() > DEADLINE:
                break
            pos = find_best_pos(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                stamp(bins[b_idx], x, y, z, w, h, d)
                placements.append((b_idx, x, y, z, w, h, d))
                placed[idx] = True
                placed_flag = True
                break

        if not placed_flag:
            b_idx = len(bins)
            hm = new_heightmap()
            bins.append(hm)
            x, y, z = 0, 0, 0
            stamp(hm, x, y, z, w, h, d)
            placements.append((b_idx, x, y, z, w, h, d))
            placed[idx] = True

    # If deadline hit, place remaining boxes each in their own new bin at origin
    for idx in range(n):
        if not placed[idx]:
            w, h, d = boxes[idx]
            b_idx = len(bins)
            hm = new_heightmap()
            bins.append(hm)
            stamp(hm, 0, 0, 0, w, h, d)
            placements.append((b_idx, 0, 0, 0, w, h, d))
            placed[idx] = True

    return placements
