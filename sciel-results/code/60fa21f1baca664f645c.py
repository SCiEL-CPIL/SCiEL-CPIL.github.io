import random
import time

_START_TIME = 0.0
_DEADLINE = 130.0  # wall-clock safety margin before the 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _has_four_collinear(points):
    """True if any 4 points are collinear (no 4-collinear constraint)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            cnt = 2
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    cnt += 1
                    if cnt >= 4:
                        return True
    return False

def _count_triples(points):
    n = len(points)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    c += 1
    return c

def _delta_triples(points, idx, old_p, new_p):
    """Change in collinear-triple count when moving point idx old_p -> new_p."""
    n = len(points)
    d = 0
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            if _is_collinear(old_p, points[i], points[j]):
                d -= 1
            if _is_collinear(new_p, points[i], points[j]):
                d += 1
    return d

def _max_3_free_subset(points):
    """Exact largest subset with no 3 collinear via bitmask branch and bound."""
    n = len(points)
    if n <= 2:
        return n
    collinear_with_pair = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_with_pair[i][j] = mask
            collinear_with_pair[j][i] = mask

    best = [0]

    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        if size + (n - i) <= best[0]:
            return
        # include point i if it does not form a collinear triple with two in subset
        can_add = True
        temp = subset_mask
        while temp:
            bit = temp & (-temp)
            a = bit.bit_length() - 1
            if collinear_with_pair[a][i] & subset_mask:
                can_add = False
                break
            temp ^= bit
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
        backtrack(i + 1, subset_mask, size)

    backtrack(0, 0, 0)
    return best[0]

def _fix_four_collinear(points):
    for _ in range(3000):
        if not _has_four_collinear(points):
            return True
        idx = random.randint(0, 19)
        old = points[idx]
        for _ in range(50):
            cand = (random.randint(-100, 100), random.randint(-100, 100))
            if cand in points or cand == old:
                continue
            points[idx] = cand
            if not _has_four_collinear(points):
                break
            points[idx] = old
        else:
            points[idx] = old
    return not _has_four_collinear(points)

def _local_search(points, budget):
    """Line-targeted local search: greedily pack disjoint collinear triples by
    moving points onto lines, then refine with random moves. Minimize the exact
    3-free subset size while keeping the no-4-collinear constraint."""
    n = len(points)
    cur_score = _max_3_free_subset(points)

    # candidate target lines: through each pair of current points
    target_lines = []
    for a in range(n):
        for b in range(a + 1, n):
            pa = points[a]
            pb = points[b]
            dx = pb[0] - pa[0]
            dy = pb[1] - pa[1]
            if dx == 0 and dy == 0:
                continue
            target_lines.append((pa, (dx, dy), a, b))
    random.shuffle(target_lines)

    for _line in range(600):
        if _time_left() < 3.0:
            break
        pa, (dx, dy), a, b = random.choice(target_lines)
        # pick a point not on this line and not a, b to move onto the line
        on_line = {a, b}
        for t in range(n):
            if t in on_line:
                continue
            if _is_collinear(pa, points[a], points[t]):
                on_line.add(t)
        movable = [t for t in range(n) if t not in on_line]
        if not movable:
            continue
        idx = random.choice(movable)
        old = points[idx]
        # try integer steps along the line for a valid (no-4, distinct) placement
        placed = False
        for s in random.sample(range(1, 14), 13):
            for sign in (1, -1):
                nx = pa[0] + sign * s * dx
                ny = pa[1] + sign * s * dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                cand = (nx, ny)
                if cand == old or cand in points:
                    continue
                points[idx] = cand
                if _has_four_collinear(points):
                    points[idx] = old
                    continue
                ns = _max_3_free_subset(points)
                if ns < cur_score:
                    cur_score = ns
                    placed = True
                    break
                points[idx] = old
            if placed:
                break

    # random-move refinement
    for _ in range(4000):
        if _time_left() < 3.0:
            break
        idx = random.randint(0, n - 1)
        old = points[idx]
        cand = (random.randint(-100, 100), random.randint(-100, 100))
        if cand == old or cand in points:
            continue
        points[idx] = cand
        if _has_four_collinear(points):
            points[idx] = old
            continue
        ns = _max_3_free_subset(points)
        if ns < cur_score:
            cur_score = ns
        else:
            points[idx] = old
    return cur_score

def _random_points():
    pts = []
    while len(pts) < 20:
        p = (random.randint(-15, 15), random.randint(-15, 15))
        if p not in pts:
            pts.append(p)
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(7)

    best_score = 21
    best_points = None

    configs = []
    # dense random (encourages collinear triples)
    configs.append(_random_points())
    # structured grid
    g = []
    for x in range(5):
        for y in range(4):
            g.append((x * 6 - 12, y * 6 - 9))
    configs.append(g)
    # staircase
    s = []
    for i in range(20):
        s.append(((i % 6) * 5 - 10, (i // 6) * 6 - 6))
    configs.append(s)
    # second random seed
    r2 = []
    while len(r2) < 20:
        p = (random.randint(-12, 12), random.randint(-12, 12))
        if p not in r2:
            r2.append(p)
    configs.append(r2)

    for cfg in configs:
        if _time_left() < 20.0:
            break
        pts = [list(p) for p in cfg]
        if not _fix_four_collinear(pts):
            continue
        sc = _local_search(pts, None)
        if sc < best_score:
            best_score = sc
            best_points = [list(p) for p in pts]

    if best_points is None:
        # fallback: parabola (no 3 collinear, no 4 collinear), valid
        best_points = [[x, x * x] for x in range(-10, 10)]
        best_score = 20

    return [[int(p[0]), int(p[1])] for p in best_points]
