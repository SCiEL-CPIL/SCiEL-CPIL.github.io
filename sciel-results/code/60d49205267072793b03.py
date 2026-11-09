"""Structured seeds + incremental local search (single-point + two-point swap)
for 16 integer points maximizing exact-4 lines, no five collinear."""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 112.0
_BOUND = 1000


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _build_line_counts(points):
    """Return dict: line_key -> number of pairs on that line."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    return counts


def _score_from_counts(counts):
    """Return (n4, n5, n3, n2) from pair-count dict."""
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


def _valid(counts):
    return all(c < 10 for c in counts.values())


def _eval_move(points, counts, i, new_pt):
    """Evaluate moving point i to new_pt. Returns (counts_new, n4, n5, n3, n2, valid)."""
    old_pt = points[i]
    n = len(points)
    # Remove pairs involving i
    for j in range(n):
        if j == i:
            continue
        k = _line_key(old_pt, points[j])
        counts[k] -= 1
        if counts[k] == 0:
            del counts[k]
    # Add pairs with new_pt
    for j in range(n):
        if j == i:
            continue
        k = _line_key(new_pt, points[j])
        counts[k] = counts.get(k, 0) + 1
    n4, n5, n3, n2 = _score_from_counts(counts)
    ok = _valid(counts)
    # Revert
    for j in range(n):
        if j == i:
            continue
        k = _line_key(new_pt, points[j])
        counts[k] -= 1
        if counts[k] == 0:
            del counts[k]
    for j in range(n):
        if j == i:
            continue
        k = _line_key(old_pt, points[j])
        counts[k] = counts.get(k, 0) + 1
    return n4, n5, n3, n2, ok


def _eval_swap(points, counts, i, j, new_i, new_j):
    """Evaluate swapping: point i -> new_i, point j -> new_j."""
    old_i, old_j = points[i], points[j]
    n = len(points)
    # Remove pairs involving i or j (careful not to double-remove the i-j pair)
    for k in range(n):
        if k == i or k == j:
            continue
        ki = _line_key(old_i, points[k])
        counts[ki] -= 1
        if counts[ki] == 0:
            del counts[ki]
        kj = _line_key(old_j, points[k])
        counts[kj] -= 1
        if counts[kj] == 0:
            del counts[kj]
    # Remove i-j pair
    kij = _line_key(old_i, old_j)
    counts[kij] -= 1
    if counts[kij] == 0:
        del counts[kij]

    # Add pairs for new_i
    for k in range(n):
        if k == i or k == j:
            continue
        ki = _line_key(new_i, points[k])
        counts[ki] = counts.get(ki, 0) + 1
    # Add pairs for new_j
    for k in range(n):
        if k == i or k == j:
            continue
        kj = _line_key(new_j, points[k])
        counts[kj] = counts.get(kj, 0) + 1
    # Add new_i - new_j pair
    knij = _line_key(new_i, new_j)
    counts[knij] = counts.get(knij, 0) + 1

    n4, n5, n3, n2 = _score_from_counts(counts)
    ok = _valid(counts)

    # Revert
    for k in range(n):
        if k == i or k == j:
            continue
        ki = _line_key(new_i, points[k])
        counts[ki] -= 1
        if counts[ki] == 0:
            del counts[ki]
        kj = _line_key(new_j, points[k])
        counts[kj] -= 1
        if counts[kj] == 0:
            del counts[kj]
    knij2 = _line_key(new_i, new_j)
    counts[knij2] -= 1
    if counts[knij2] == 0:
        del counts[knij2]
    for k in range(n):
        if k == i or k == j:
            continue
        ki = _line_key(old_i, points[k])
        counts[ki] = counts.get(ki, 0) + 1
        kj = _line_key(old_j, points[k])
        counts[kj] = counts.get(kj, 0) + 1
    kij2 = _line_key(old_i, old_j)
    counts[kij2] = counts.get(kij2, 0) + 1

    return n4, n5, n3, n2, ok


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    """Staggered 4x4: offset every other row."""
    pts = []
    for y in range(4):
        off = 1 if y % 2 == 1 else 0
        for x in range(4):
            pts.append([x + off, y])
    return pts


def _seed_hexagonal():
    """Hexagonal-like packing of 16 points."""
    pts = []
    # Row 0: 4 points
    for x in range(4):
        pts.append([2 * x, 0])
    # Row 1: 4 points offset
    for x in range(4):
        pts.append([2 * x + 1, 3])
    # Row 2: 4 points
    for x in range(4):
        pts.append([2 * x, 6])
    # Row 3: 4 points offset
    for x in range(4):
        pts.append([2 * x + 1, 9])
    return pts


def _seed_diagonal():
    """Diagonal grid: points on a slanted lattice."""
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_perturbed():
    """4x4 grid with corner perturbations."""
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_affine():
    """Affine transform of grid: (2x, x+y) style."""
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([2 * x + y, x + 2 * y])
    return pts


def _local_search(points, rng, time_budget):
    n = len(points)
    cur = [list(p) for p in points]
    counts = _build_line_counts(cur)
    n4, n5, n3, n2 = _score_from_counts(counts)
    cur_key = (n4, -n5, n3, n2)
    best_key = cur_key
    best_pts = [list(p) for p in cur]
    best_counts = dict(counts)

    step = 0
    stall = 0
    while time.time() < time_budget:
        improved = False
        # --- Single-point moves ---
        for i in range(n):
            if time.time() >= time_budget:
                return best_pts
            x0, y0 = cur[i]
            w = 4 + min(step // 15, 8)
            best_local_key = None
            best_local_pt = None
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    m4, m5, m3, m2, ok = _eval_move(cur, counts, i, np_)
                    if not ok:
                        continue
                    mk = (m4, -m5, m3, m2)
                    if best_local_key is None or mk > best_local_key:
                        best_local_key = mk
                        best_local_pt = np_
            if best_local_pt is not None and best_local_key > cur_key:
                cur[i] = best_local_pt
                # Update counts in-place
                old_pt = [x0, y0]
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(old_pt, cur[j])
                    counts[k] -= 1
                    if counts[k] == 0:
                        del counts[k]
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(best_local_pt, cur[j])
                    counts[k] = counts.get(k, 0) + 1
                n4, n5, n3, n2 = _score_from_counts(counts)
                cur_key = (n4, -n5, n3, n2)
                improved = True
                if cur_key > best_key:
                    best_key = cur_key
                    best_pts = [list(p) for p in cur]
                    best_counts = dict(counts)
        # --- Two-point swap moves ---
        if time.time() < time_budget - 0.5:
            for _ in range(3):
                if time.time() >= time_budget:
                    break
                i = rng.randrange(n)
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1
                xi, yi = cur[i]
                xj, yj = cur[j]
                found = False
                for _ in range(80):
                    nxi = xi + rng.randint(-5, 5)
                    nyi = yi + rng.randint(-5, 5)
                    nxj = xj + rng.randint(-5, 5)
                    nyj = yj + rng.randint(-5, 5)
                    if abs(nxi) > _BOUND or abs(nyi) > _BOUND:
                        continue
                    if abs(nxj) > _BOUND or abs(nyj) > _BOUND:
                        continue
                    new_i = [nxi, nyi]
                    new_j = [nxj, nyj]
                    if any(new_i == p or new_j == p for p in cur):
                        continue
                    if new_i == new_j:
                        continue
                    m4, m5, m3, m2, ok = _eval_swap(cur, counts, i, j, new_i, new_j)
                    if not ok:
                        continue
                    mk = (m4, -m5, m3, m2)
                    if mk > cur_key:
                        # Apply swap
                        cur[i] = new_i
                        cur[j] = new_j
                        n4, n5, n3, n2 = m4, m5, m3, m2
                        cur_key = mk
                        improved = True
                        if cur_key > best_key:
                            best_key = cur_key
                            best_pts = [list(p) for p in cur]
                            best_counts = dict(counts)
                        found = True
                        break
                if not found:
                    # Try a random kick (accept equal or slightly worse)
                    for _ in range(30):
                        nxi = xi + rng.randint(-6, 6)
                        nyi = yi + rng.randint(-6, 6)
                        if abs(nxi) > _BOUND or abs(nyi) > _BOUND:
                            continue
                        new_i = [nxi, nyi]
                        if any(new_i == p for p in cur):
                            continue
                        m4, m5, m3, m2, ok = _eval_move(cur, counts, i, new_i)
                        if not ok:
                            continue
                        mk = (m4, -m5, m3, m2)
                        if mk >= cur_key:
                            cur[i] = new_i
                            old_pt = [xi, yi]
                            for k in range(n):
                                if k == i:
                                    continue
                                kk = _line_key(old_pt, cur[k])
                                counts[kk] -= 1
                                if counts[kk] == 0:
                                    del counts[kk]
                            for k in range(n):
                                if k == i:
                                    continue
                                kk = _line_key(new_i, cur[k])
                                counts[kk] = counts.get(kk, 0) + 1
                            n4, n5, n3, n2 = m4, m5, m3, m2
                            cur_key = mk
                            improved = True
                            if cur_key > best_key:
                                best_key = cur_key
                                best_pts = [list(p) for p in cur]
                                best_counts = dict(counts)
                            break
        step += 1
        if improved:
            stall = 0
        else:
            stall += 1
            if stall > 8:
                # Bigger random restart of one point
                i = rng.randrange(n)
                x0, y0 = cur[i]
                for _ in range(500):
                    nx = x0 + rng.randint(-10, 10)
                    ny = y0 + rng.randint(-10, 10)
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    m4, m5, m3, m2, ok = _eval_move(cur, counts, i, np_)
                    if not ok:
                        continue
                    mk = (m4, -m5, m3, m2)
                    if mk >= cur_key:
                        cur[i] = np_
                        old_pt = [x0, y0]
                        for k in range(n):
                            if k == i:
                                continue
                            kk = _line_key(old_pt, cur[k])
                            counts[kk] -= 1
                            if counts[kk] == 0:
                                del counts[kk]
                        for k in range(n):
                            if k == i:
                                continue
                            kk = _line_key(np_, cur[k])
                            counts[kk] = counts.get(kk, 0) + 1
                        n4, n5, n3, n2 = m4, m5, m3, m2
                        cur_key = mk
                        if cur_key > best_key:
                            best_key = cur_key
                            best_pts = [list(p) for p in cur]
                            best_counts = dict(counts)
                        break
    return best_pts


def construct():
    rng = random.Random(42)
    seeds = [
        _seed_grid(),
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_diagonal(),
        _seed_perturbed(),
        _seed_affine(),
    ]
    # Add a couple random seeds
    for _ in range(2):
        pts = []
        for _ in range(16):
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 16:
            x = rng.randint(-15, 15)
            y = rng.randint(-15, 15)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)

    # Deduplicate
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    # Filter to valid seeds
    valid_seeds = []
    for s in uniq:
        c = _build_line_counts(s)
        if _valid(c):
            valid_seeds.append(s)

    if not valid_seeds:
        valid_seeds = [uniq[0]]

    best_pts = None
    best_key = None
    total_time = _DEADLINE - time.time()
    # Allocate time: more to better seeds
    # First pass: quick evaluation of all seeds
    seed_scores = []
    for s in valid_seeds:
        c = _build_line_counts(s)
        n4, n5, n3, n2 = _score_from_counts(c)
        seed_scores.append((n4, -n5, n3, n2, s))
    seed_scores.sort(key=lambda x: x[:4], reverse=True)

    # Allocate time proportional to rank (best gets more)
    n_seeds = len(valid_seeds)
    # Give top seed 40%, next 25%, rest split
    budgets = []
    remaining = total_time
    for idx, sc in enumerate(seed_scores):
        if idx == 0:
            b = remaining * 0.35
        elif idx == 1:
            b = remaining * 0.25
        elif idx == 2:
            b = remaining * 0.20
        else:
            b = remaining * (0.20 / max(1, n_seeds - 2))
        b = max(b, 2.0)
        b = min(b, remaining - 1.0)
        budgets.append(b)
        remaining -= b
        if remaining < 3.0:
            budgets[-1] = remaining
            break

    for idx, sc in enumerate(seed_scores):
        if time.time() >= _DEADLINE - 2.0:
            break
        s = sc[4]
        b = budgets[idx] if idx < len(budgets) else 3.0
        t_end = min(_DEADLINE - 1.5, time.time() + b)
        if t_end <= time.time():
            break
        pts = _local_search(s, rng, t_end)
        if pts is not None:
            c = _build_line_counts(pts)
            n4, n5, n3, n2 = _score_from_counts(c)
            k = (n4, -n5, n3, n2)
            if best_key is None or k > best_key:
                best_key = k
                best_pts = pts

    # If we still have time, do another round on the best
    if time.time() < _DEADLINE - 5.0 and best_pts is not None:
        t_end = _DEADLINE - 1.5
        pts = _local_search(best_pts, rng, t_end)
        if pts is not None:
            c = _build_line_counts(pts)
            n4, n5, n3, n2 = _score_from_counts(c)
            k = (n4, -n5, n3, n2)
            if k > best_key:
                best_key = k
                best_pts = pts

    # Final safety
    if best_pts is None:
        return _seed_grid()

    c = _build_line_counts(best_pts)
    if not _valid(c):
        return _seed_grid()
    for p in best_pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    if len(set(map(tuple, best_pts))) != 16:
        return _seed_grid()
    return [list(p) for p in best_pts]
