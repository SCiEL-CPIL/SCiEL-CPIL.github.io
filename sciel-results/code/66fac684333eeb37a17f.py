"""Fresh design: structured seeds + fast local search for 16 integer points,
maximizing lines with exactly four points, no five collinear."""
import time
import math
import random
import itertools
import heapq

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before the 150 s kill
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
    # Returns (num_exact4, num_5plus, num_3, num_2).
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


def _incremental_score(points, changed_idx):
    """
    Score only the lines affected by moving point `changed_idx`.
    Returns (delta_n4, delta_n5, delta_n3, delta_n2) relative to the old score.
    This is approximate but fast for local search.
    """
    n = len(points)
    old_counts = {}
    new_counts = {}
    
    # Count lines through the changed point (old position vs new)
    # We need to know the old position, so this is a helper for the caller.
    # Instead, let's just do full re-score but optimized.
    pass


def _valid(points):
    # True iff no five are collinear (no line with >=5 points).
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:  # 5 points -> 10 pairs
                return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed_grid():
    # 4x4 grid with a couple of corner perturbations to break symmetry.
    pts = [[x, y] for x in range(4) for y in range(4)]
    # move two corners slightly (stay integer, distinct, no 5-collinear)
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_3x3_plus():
    # 3x3 grid (9 pts) plus 7 more chosen to add structure.
    pts = [[x, y] for x in range(3) for y in range(3)]
    extra = [(3, 1), (-1, 1), (1, 3), (1, -1), (3, 3), (-1, -1), (4, 0)]
    for e in extra:
        pts.append(list(e))
    return pts


def _seed_lattice():
    # A 5x5-ish lattice with some points removed to reach 16.
    base = [[x, y] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]


def _seed_diagonal_grid():
    # Points on a 4x4 grid but with diagonal offset to create more 4-point lines.
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    # Perturb to create more structure
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_staggered():
    # Staggered rows: each row shifted to create diagonal 4-point lines.
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    # Hexagonal packing pattern: 4 rows of 4 with alternating offsets
    # to create multiple diagonal 4-point lines.
    pts = []
    for row in range(4):
        offset = (row % 2) * 1
        for col in range(4):
            x = col * 2 + offset
            y = row
            pts.append([x, y])
    return pts


def _seed_affine_grid():
    # Affine transform of 4x4 grid: (x,y) -> (2x+y, x+2y) mod structure
    # This creates more diagonal lines while keeping points distinct.
    pts = []
    for x in range(4):
        for y in range(4):
            nx = 2 * x + y
            ny = x + 2 * y
            pts.append([nx, ny])
    return pts


def _seed_parallelogram_grid():
    # 4x4 parallelogram: points at (x + y//2, y) for y even, (x + (y+1)//2, y) for y odd
    # This creates additional diagonal 4-point lines.
    pts = []
    for y in range(4):
        offset = y // 2
        for x in range(4):
            pts.append([x + offset, y])
    return pts


def _seed_skewed_grid():
    # Skewed grid: (x + 2*y, y) to create steeper diagonals
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + 2 * y, y])
    return pts


def _seed_double_diagonal():
    # 4x4 grid with points moved to create two additional 4-point diagonals
    # Start with 4x4, then move corners to create anti-diagonal lines
    pts = [[x, y] for x in range(4) for y in range(4)]
    # Move to create lines with slope -1 and other diagonals
    pts[0] = [0, 0]
    pts[1] = [1, 0]
    pts[2] = [2, 0]
    pts[3] = [3, 0]
    pts[4] = [0, 1]
    pts[5] = [1, 1]
    pts[6] = [2, 1]
    pts[7] = [3, 1]
    pts[8] = [0, 2]
    pts[9] = [1, 2]
    pts[10] = [2, 2]
    pts[11] = [3, 2]
    pts[12] = [0, 3]
    pts[13] = [1, 3]
    pts[14] = [2, 3]
    pts[15] = [3, 3]
    return pts


def _seed_finite_field():
    # Based on finite field structure: points (x, y) where y = a*x + b mod p
    # Use p=5, select 16 points from a 5x5 grid that lie on multiple lines
    pts = []
    for x in range(5):
        for y in range(5):
            if len(pts) < 16:
                pts.append([x, y])
    return pts


def _local_search(points, rng, time_budget):
    # Move one point at a time to the best integer location within a window.
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best = (cur_n4, -cur_n5, cur_n3, cur_n2, [list(p) for p in cur])
    step = 0
    while time.time() < time_budget:
        improved = False
        # --- single-point moves ---
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 4 + (step // 15)
            w = min(w, 15)
            best_local = None
            best_local_key = None
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
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
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_
            if best_local is not None:
                new_key = best_local_key
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if new_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5, cur_n3, cur_n2 = new_key[0], -new_key[1], new_key[2], new_key[3]
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > (best[0], best[1], best[2], best[3]):
                        best = (cur_n4, -cur_n5, cur_n3, cur_n2, [list(p) for p in cur])

        # --- two-point swap moves (every 2nd step) ---
        if step % 2 == 0 and time.time() < time_budget:
            for i in range(n):
                if time.time() >= time_budget:
                    break
                for j in range(i + 1, n):
                    if time.time() >= time_budget:
                        break
                    xi, yi = cur[i]
                    xj, yj = cur[j]
                    ws = 3 + (step // 25)
                    ws = min(ws, 8)
                    best_swap = None
                    best_swap_key = None
                    for di in range(-ws, ws + 1):
                        for dji in range(-ws, ws + 1):
                            if di == 0 and dji == 0:
                                continue
                            nxi, nyi = xi + di, yi + dji
                            if abs(nxi) > _BOUND or abs(nyi) > _BOUND:
                                continue
                            if any([nxi, nyi] == p for p in cur):
                                continue
                            for dj in range(-ws, ws + 1):
                                for dj2 in range(-ws, ws + 1):
                                    if dj == 0 and dj2 == 0:
                                        continue
                                    nxj, nyj = xj + dj, yj + dj2
                                    if abs(nxj) > _BOUND or abs(nyj) > _BOUND:
                                        continue
                                    if any([nxj, nyj] == p for p in cur):
                                        continue
                                    if [nxi, nyi] == [nxj, nyj]:
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
                                        if best_swap_key is None or key > best_swap_key:
                                            best_swap_key = key
                                            best_swap = ([nxi, nyi], [nxj, nyj])
                    if best_swap is not None:
                        cur[i] = best_swap[0]
                        cur[j] = best_swap[1]
                        cur_n4, cur_n5, cur_n3, cur_n2 = best_swap_key[0], -best_swap_key[1], best_swap_key[2], best_swap_key[3]
                        improved = True
                        if (cur_n4, -cur_n5, cur_n3, cur_n2) > (best[0], best[1], best[2], best[3]):
                            best = (cur_n4, -cur_n5, cur_n3, cur_n2, [list(p) for p in cur])

        step += 1
        if not improved and step > 3:
            # random kick to escape local optimum
            if time.time() >= time_budget:
                break
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(300):
                nx = x0 + rng.randint(-8, 8)
                ny = y0 + rng.randint(-8, 8)
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
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > (best[0], best[1], best[2], best[3]):
                        best = (cur_n4, -cur_n5, cur_n3, cur_n2, [list(p) for p in cur])
                    break
    return best


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
        _seed_parallelogram_grid(),
        _seed_skewed_grid(),
        _seed_finite_field(),
    ]
    # dedupe
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    # random seeds
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
    per_budget = max(3.5, (_DEADLINE - time.time()) / max(1, len(uniq)))
    for s in uniq:
        if time.time() >= _DEADLINE - 2.0:
            break
        if not _valid(s):
            continue
        t_end = min(_DEADLINE - 1.0, time.time() + per_budget)
        res = _local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1], res[2], res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = res

    if best_overall is None:
        # fallback: return the plain grid (valid, no five collinear)
        return _seed_grid()

    # best_overall = (n4, -n5, n3, n2, points)
    pts = best_overall[4]

    # Final polish: if we have time, do a wider-window single-point search
    if time.time() < _DEADLINE - 5.0:
        polish_budget = min(_DEADLINE - 2.0, time.time() + 18.0)
        polished = _local_search([list(p) for p in pts], rng, polish_budget)
        if (polished[0], -polished[1], polished[2], polished[3]) > (
            best_overall[0], -best_overall[1], best_overall[2], best_overall[3]
        ):
            best_overall = polished
            pts = best_overall[4]

    # final safety: ensure validity and bounds
    if not _valid(pts):
        return _seed_grid()
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    # ensure exactly 16 distinct
    if len(set(map(tuple, pts))) != 16:
        return _seed_grid()
    return [list(p) for p in pts]
