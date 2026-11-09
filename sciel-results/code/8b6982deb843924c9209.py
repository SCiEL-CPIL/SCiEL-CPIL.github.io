"""Combined program: incremental line-count scoring + local search (single moves, 2-swaps, restarts)."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _line_key(p1, p2):
    """Canonical integer key for the (undirected) line through two distinct points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # line: a*x + b*y = c  with (a,b) = (dy, -dx), normalized
    a = dy
    b = -dx
    g2 = math.gcd(abs(a), abs(b))
    a //= g2
    b //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * p1[0] + b * p1[1]
    return (a, b, c)

def _score(points):
    """Return (num_4point_lines, num_5plus_lines)."""
    from collections import Counter
    cnt = Counter()
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k is not None:
                cnt[k] += 1
    c4 = 0
    c5 = 0
    for c in cnt.values():
        if c == 4:
            c4 += 1
        elif c >= 5:
            c5 += 1
    return c4, c5

def construct():
    deadline = time.time() + 135
    rng = random.Random(12345)
    RANGE = 1000

    def valid_distinct(pts):
        seen = set()
        for p in pts:
            x, y = p
            if abs(x) > RANGE or abs(y) > RANGE:
                return False
            if (x, y) in seen:
                return False
            seen.add((x, y))
        return True

    def build_map(points):
        from collections import Counter
        cnt = Counter()
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                k = _line_key(points[i], points[j])
                if k is not None:
                    cnt[k] += 1
        return cnt

    def c4c5_from_map(cnt):
        c4 = 0
        c5 = 0
        for c in cnt.values():
            if c == 4:
                c4 += 1
            elif c >= 5:
                c5 += 1
        return c4, c5

    def objective(c4, c5):
        return c4 - 1000 * c5

    # ---- starting population ----
    starts = []
    # 4x4 grid (canonical seed)
    starts.append([[x, y] for x in range(4) for y in range(4)])
    # a few deterministic structured starts
    starts.append([[x, y] for x in range(4) for y in range(4, 8)])
    starts.append([[x, y] for x in range(-3, 1) for y in range(-3, 1)])
    # random starts
    for seed in (7, 21, 99, 313, 777, 1234):
        r = random.Random(seed)
        pts = []
        seen = set()
        while len(pts) < 16:
            x = r.randint(-RANGE, RANGE)
            y = r.randint(-RANGE, RANGE)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            pts.append([x, y])
        starts.append(pts)

    best_obj = -10**9
    best_points = None

    def local_search(start_pts, deadline):
        nonlocal best_obj, best_points
        points = [p[:] for p in start_pts]
        if not valid_distinct(points):
            return
        cnt = build_map(points)
        c4, c5 = c4c5_from_map(cnt)
        obj = objective(c4, c5)
        if obj > best_obj:
            best_obj = obj
            best_points = [p[:] for p in points]

        n = len(points)
        # single-point move
        while time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                ox, oy = points[i]
                # small neighbourhood first, then larger
                for radius in (1, 2, 3, 5, 8, 13, 21, 34, 55):
                    if time.time() >= deadline:
                        break
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx = ox + dx
                            ny = oy + dy
                            if abs(nx) > RANGE or abs(ny) > RANGE:
                                continue
                            # distinctness
                            if any(points[j][0] == nx and points[j][1] == ny for j in range(n) if j != i):
                                continue
                            # incremental update
                            old = (ox, oy)
                            new = (nx, ny)
                            removed = {}
                            for j in range(n):
                                if j == i:
                                    continue
                                k = _line_key(old, points[j])
                                if k is None:
                                    continue
                                cnt[k] -= 1
                                removed[k] = removed.get(k, 0) + 1
                            added = {}
                            for j in range(n):
                                if j == i:
                                    continue
                                k = _line_key(new, points[j])
                                if k is None:
                                    continue
                                cnt[k] = cnt.get(k, 0) + 1
                                added[k] = added.get(k, 0) + 1
                            c4, c5 = c4c5_from_map(cnt)
                            nobj = objective(c4, c5)
                            if nobj > obj:
                                points[i] = [nx, ny]
                                obj = nobj
                                improved = True
                                if obj > best_obj:
                                    best_obj = obj
                                    best_points = [p[:] for p in points]
                                # restart neighbourhood from improved point
                                break
                            else:
                                # revert
                                for k, v in added.items():
                                    cnt[k] -= v
                                    if cnt[k] == 0:
                                        del cnt[k]
                                for k, v in removed.items():
                                    cnt[k] = cnt.get(k, 0) + v
                                points[i] = [ox, oy]
                        else:
                            continue
                        break
            # 2-point swap move (helps break 5-collinear / create new 4-lines)
            if time.time() < deadline:
                for _ in range(200):
                    if time.time() >= deadline:
                        break
                    i = rng.randrange(n)
                    j = rng.randrange(n)
                    if i == j:
                        continue
                    oi = points[i][:]
                    oj = points[j][:]
                    # candidate: move i and j to random new spots
                    ni = [rng.randint(-RANGE, RANGE), rng.randint(-RANGE, RANGE)]
                    nj = [rng.randint(-RANGE, RANGE), rng.randint(-RANGE, RANGE)]
                    if (ni[0] == oi[0] and ni[1] == oi[1]) or (nj[0] == oj[0] and nj[1] == oj[1]):
                        continue
                    # distinctness
                    cand = points[:]
                    cand[i] = ni
                    cand[j] = nj
                    if not valid_distinct(cand):
                        continue
                    # incremental: remove lines through i and j, add new ones
                    saved = {}
                    def remove_pt(p):
                        for j2 in range(n):
                            if j2 == i or j2 == j:
                                continue
                            k = _line_key(p, cand[j2])
                            if k is None:
                                continue
                            cnt[k] = cnt.get(k, 0) - 1
                            saved[k] = saved.get(k, 0) + 1
                    remove_pt(oi)
                    remove_pt(oj)
                    def add_pt(p):
                        for j2 in range(n):
                            if j2 == i or j2 == j:
                                continue
                            k = _line_key(p, cand[j2])
                            if k is None:
                                continue
                            cnt[k] = cnt.get(k, 0) + 1
                    add_pt(ni)
                    add_pt(nj)
                    c4, c5 = c4c5_from_map(cnt)
                    nobj = objective(c4, c5)
                    if nobj > obj:
                        obj = nobj
                        improved = True
                        if obj > best_obj:
                            best_obj = obj
                            best_points = [p[:] for p in cand]
                    else:
                        # revert
                        for k, v in saved.items():
                            cnt[k] = cnt.get(k, 0) + v
                        for j2 in range(n):
                            if j2 == i or j2 == j:
                                continue
                            k = _line_key(ni, cand[j2])
                            if k is not None:
                                cnt[k] -= 1
                            k = _line_key(nj, cand[j2])
                            if k is not None:
                                cnt[k] -= 1
                        points[i] = oi
                        points[j] = oj
                    if improved:
                        break
            if not improved:
                # stuck: break out to a new restart
                break

    for s in starts:
        if time.time() >= deadline:
            break
        local_search(s, deadline)

    # final safety: ensure a valid configuration
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]

    # verify no 5-collinear; if somehow invalid, fall back to grid
    c4, c5 = _score(best_points)
    if c5 > 0:
        best_points = [[x, y] for x in range(4) for y in range(4)]
    return best_points
# EVOLVE-BLOCK-END
