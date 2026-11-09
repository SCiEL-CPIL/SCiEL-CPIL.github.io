"""Combined approach: structured seeds + fast incremental local search with
precomputed line keys, targeted candidate generation, and two-point moves.
Maximizes lines with exactly four points, no five collinear."""
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 115.0
_BOUND = 1000


def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score(points):
    """Returns (num_exact4, num_5plus, num_3, num_2)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = n3 = n2 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
        elif c == 2:
            n2 += 1
    return n4, n5, n3, n2


def _valid(points):
    """True iff no five are collinear."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed_grid():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_3x3_plus():
    pts = [[x, y] for x in range(3) for y in range(3)]
    extra = [(3, 1), (-1, 1), (1, 3), (1, -1), (3, 3), (-1, -1), (4, 0)]
    for e in extra:
        pts.append(list(e))
    return pts


def _seed_lattice():
    base = [[x, y] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]


def _seed_diagonal_grid():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        pts.append([dx, dy])
    ring2 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, -1), (1, 2), (-1, -2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _seed_affine_grid():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_mixed():
    pts = []
    for i in range(4):
        for j in range(4):
            if i % 2 == 0:
                pts.append([i, j])
            else:
                pts.append([i + 1, j])
    return pts


def _seed_sheared():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + 2 * y, y])
    return pts


def _fast_local_search(points, rng, time_budget):
    """Fast local search with precomputed line keys and incremental updates."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]
    step = 0

    while time.time() < time_budget:
        improved = False

        # Precompute all line keys for current config
        line_keys = {}
        for i in range(n):
            for j in range(i + 1, n):
                k = _line_key(cur[i], cur[j])
                line_keys[(i, j)] = k

        # Count points per line
        line_counts = {}
        for (i, j), k in line_keys.items():
            line_counts[k] = line_counts.get(k, 0) + 1

        # For each point, find lines through it and their counts
        point_lines = {}
        for i in range(n):
            pl = []
            for j in range(n):
                if j != i:
                    key = (min(i, j), max(i, j))
                    k = line_keys.get(key)
                    if k is None:
                        k = _line_key(cur[i], cur[j])
                        line_keys[key] = k
                    cnt = line_counts.get(k, 0)
                    pl.append((j, k, cnt))
            point_lines[i] = pl

        # Try moving each point
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 6 + (step // 8)
            w = min(w, 25)

            best_local = None
            best_local_key = None

            # Generate candidate positions
            candidates = []
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    # Check distinctness
                    if any(np_ == p for p in cur):
                        continue
                    candidates.append(np_)

            # Also add some targeted candidates: positions that might create new 4-lines
            # For each other point j, consider positions that would align with 2 other points
            for j in range(n):
                if j == i:
                    continue
                for m in range(n):
                    if m == i or m == j:
                        continue
                    # Position that would make i, j, m collinear with a 4th point
                    # We'll just use a few random offsets
                    for _ in range(3):
                        nx = cur[j][0] + rng.randint(-15, 15)
                        ny = cur[j][1] + rng.randint(-15, 15)
                        if abs(nx) > _BOUND or abs(ny) > _BOUND:
                            continue
                        np_ = [nx, ny]
                        if not any(np_ == p for p in cur):
                            candidates.append(np_)

            # Limit candidates to keep it fast
            if len(candidates) > 200:
                candidates = candidates[:200]

            for np_ in candidates:
                if time.time() >= time_budget:
                    break

                # Fast validity check: for each line through new position,
                # count how many other points are on it
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    # Count points on this line (excluding i and j)
                    cnt = 1  # j is on the line
                    for m in range(n):
                        if m == i or m == j:
                            continue
                        if _line_key(np_, cur[m]) == k:
                            cnt += 1
                    if cnt >= 5:
                        valid = False
                        break
                if not valid:
                    continue

                # Score the candidate
                cand = [list(p) for p in cur]
                cand[i] = np_
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                if best_local_key is None or key > best_local_key:
                    best_local_key = key
                    best_local = np_

            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5, cur_n3, cur_n2 = best_local_key[0], -best_local_key[1], best_local_key[2], best_local_key[3]
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]

        # Two-point moves
        if not improved or step % 3 == 0:
            if time.time() >= time_budget:
                break
            for _ in range(30):
                if time.time() >= time_budget:
                    break
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                nxi = cur[i][0] + rng.randint(-8, 8)
                nyi = cur[i][1] + rng.randint(-8, 8)
                nxj = cur[j][0] + rng.randint(-8, 8)
                nyj = cur[j][1] + rng.randint(-8, 8)
                if abs(nxi) > _BOUND or abs(nyi) > _BOUND or abs(nxj) > _BOUND or abs(nyj) > _BOUND:
                    continue
                if [nxi, nyi] == [nxj, nyj]:
                    continue
                if any([nxi, nyi] == p for p in cur) or any([nxj, nyj] == p for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = [nxi, nyi]
                cand[j] = [nxj, nyj]
                if not _valid(cand):
                    continue
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if key > cur_key:
                    cur[i] = [nxi, nyi]
                    cur[j] = [nxj, nyj]
                    cur_n4, cur_n5, cur_n3, cur_n2 = n4, n5, n3, n2
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]

        step += 1

        if not improved and step > 4:
            # Random kick
            if time.time() >= time_budget:
                break
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(800):
                nx = x0 + rng.randint(-12, 12)
                ny = y0 + rng.randint(-12, 12)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                if not _valid(cand):
                    continue
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if key >= cur_key:
                    cur[i] = np_
                    cur_n4, cur_n5, cur_n3, cur_n2 = n4, n5, n3, n2
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]
                    break

    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)


def _polish(points, rng, time_budget):
    """Final polish with wider search window."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]

    for i in range(n):
        if time.time() >= time_budget:
            break
        x0, y0 = cur[i]
        w = 35
        best_local = None
        best_local_key = None

        for dx in range(-w, w + 1):
            if time.time() >= time_budget:
                break
            for dy in range(-w, w + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue

                # Fast validity
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    cnt = 1
                    for m in range(n):
                        if m == i or m == j:
                            continue
                        if _line_key(np_, cur[m]) == k:
                            cnt += 1
                    if cnt >= 5:
                        valid = False
                        break
                if not valid:
                    continue

                cand = [list(p) for p in cur]
                cand[i] = np_
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                if best_local_key is None or key > best_local_key:
                    best_local_key = key
                    best_local = np_

        if best_local is not None:
            cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
            if best_local_key > cur_key:
                cur[i] = best_local
                cur_n4, cur_n5, cur_n3, cur_n2 = best_local_key[0], -best_local_key[1], best_local_key[2], best_local_key[3]
                if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                    best_pts = [list(p) for p in cur]

    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)


def construct():
    global _START, _DEADLINE
    rng = random.Random(12345)

    seeds = [
        _seed_grid(),
        _seed_perturbed_grid(),
        _seed_3x3_plus(),
        _seed_lattice(),
        _seed_diagonal_grid(),
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_affine_grid(),
        _seed_mixed(),
        _seed_sheared(),
    ]

    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    # Add a few random seeds
    for _ in range(4):
        pts = []
        for _ in range(16):
            x = rng.randint(-15, 15)
            y = rng.randint(-15, 15)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 16:
            x = rng.randint(-25, 25)
            y = rng.randint(-25, 25)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        key = tuple(sorted(map(tuple, pts)))
        if key not in seen:
            seen.add(key)
            uniq.append(pts)

    best_overall = None
    # Allocate time: give more to early structured seeds
    remaining = _DEADLINE - time.time()
    per_budget = max(3.0, remaining / max(1, len(uniq)))

    for idx, s in enumerate(uniq):
        if time.time() >= _DEADLINE - 5.0:
            break
        if not _valid(s):
            continue
        if idx < 8:
            t_end = min(_DEADLINE - 4.0, time.time() + per_budget * 1.3)
        else:
            t_end = min(_DEADLINE - 4.0, time.time() + per_budget)
        res = _fast_local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1], res[2], res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = res

    # Final polish
    if best_overall is not None and time.time() < _DEADLINE - 5.0:
        pts = best_overall[4]
        polish_budget = min(_DEADLINE - 3.0, time.time() + 20.0)
        polish_res = _polish(pts, rng, polish_budget)
        if (polish_res[0], -polish_res[1], polish_res[2], polish_res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = polish_res

    if best_overall is None:
        return _seed_grid()

    pts = best_overall[4]

    if not _valid(pts):
        return _seed_grid()
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    if len(set(map(tuple, pts))) != 16:
        return _seed_grid()

    return [list(p) for p in pts]
