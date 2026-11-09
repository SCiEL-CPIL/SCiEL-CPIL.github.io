import time
import math
import random
from itertools import combinations

_DEADLINE = time.time() + 112.0   # explicit wall-clock guard (150s kill -> safety margin)
_BOUND = 1000


def _gcd(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a


# ---- dual-plane representation -------------------------------------------
# A point (a,b) in the primal plane maps to the dual line  a*x + b*y = 1.
# A primal line through k points  <=>  k concurrent dual lines.
#   "no five collinear"  <=>  "no five concurrent dual lines"
#   "exactly 4 collinear" <=> "exactly 4 concurrent dual lines"
# We build the dual intersection graph and use it for fast validity checks.


def _line_key(p, q):
    """Canonical key of the (primal) line through points p,q."""
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


def _build_counts(points):
    """Map line_key -> number of points on that line."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    return counts


def _build_graph(points, counts):
    """Dual intersection graph: graph[i][j] = line_key of line through i,j."""
    n = len(points)
    graph = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            graph[i][j] = graph[j][i] = k
    return graph


def _exact4(counts):
    return sum(1 for c in counts.values() if c == 4)


def _validity(points, graph):
    """True iff no five points are collinear (no line with >=5 points).
    Uses the dual intersection graph for a fast early-exit check."""
    n = len(points)
    for i in range(n):
        seen = {}
        for j in range(n):
            if i == j:
                continue
            k = graph[i][j]
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 4:
                return False   # point i + 4 others on one line => 5 collinear
    return True


def _score(points, graph):
    counts = _build_counts(points)
    return _exact4(counts), counts


# ---- seeds ----------------------------------------------------------------
def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    pts = []
    for row in range(4):
        for col in range(4):
            pts.append([col + row, row])
    return pts


def _seed_affine():
    # shear (x,y)->(x+y,y): keeps 4-point rows/cols, adds diagonal structure
    return [[x + y, y] for x in range(4) for y in range(4)]


def _seed_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        pts.append([dx, dy])
    for p in [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, -1), (1, 2), (-1, -2)]:
        pts.append(list(p))
    return pts[:16]


def _seed_lattice():
    base = [[x, y] for x in range(6) for y in range(6)]
    random.shuffle(base)
    return base[:16]


def _seed_random():
    pts = set()
    while len(pts) < 16:
        pts.add((random.randint(-8, 8), random.randint(-8, 8)))
    return [list(p) for p in pts]


# ---- optimization ---------------------------------------------------------
def _optimize(points, rng, budget):
    """Hill-climb (best-improving single-point moves) with annealing kicks."""
    n = len(points)
    cur = [list(p) for p in points]
    graph = _build_graph(cur, None)
    counts = _build_counts(cur)
    cur_score = _exact4(counts)
    valid = _validity(cur, graph)

    best_score = cur_score if valid else -10**9
    best_pts = [list(p) for p in cur]

    T = 8.0
    T_min = 0.05
    cooling = 0.985
    stagnation = 0

    while time.time() < budget:
        improved = False
        best_move = None
        best_move_score = cur_score

        for i in range(n):
            x0, y0 = cur[i]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue

                    # --- incremental validity: only lines through i change ---
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        cnt = 1
                        for m in range(n):
                            if m == i or m == j:
                                continue
                            if graph[m][j] == k:
                                cnt += 1
                        if cnt >= 4:   # i + j + 3 others => 5 collinear
                            ok = False
                            break
                    if not ok:
                        continue

                    # --- incremental exact-4 delta ---
                    delta = 0
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        old_c = counts.get(k, 2)
                        new_c = old_c - 1
                        if old_c == 4:
                            delta -= 1
                        if new_c == 4:
                            delta += 1
                    cand_score = cur_score + delta
                    if cand_score > best_move_score:
                        best_move_score = cand_score
                        best_move = (i, np_)

        if best_move is not None:
            i, np_ = best_move
            # apply
            for j in range(n):
                if j == i:
                    continue
                k = _line_key(np_, cur[j])
                counts[k] = counts.get(k, 0) + 1
                graph[i][j] = graph[j][i] = k
            cur[i] = np_
            cur_score = best_move_score
            improved = True
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur]
        else:
            stagnation += 1

        # --- simulated-annealing kick to escape plateaus ---
        if not improved or stagnation >= 1:
            stagnation = 0
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(400):
                nx = x0 + rng.randint(-10, 10)
                ny = y0 + rng.randint(-10, 10)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    cnt = 1
                    for m in range(n):
                        if m == i or m == j:
                            continue
                        if graph[m][j] == k:
                            cnt += 1
                    if cnt >= 4:
                        ok = False
                        break
                if not ok:
                    continue
                delta = 0
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    old_c = counts.get(k, 2)
                    new_c = old_c - 1
                    if old_c == 4:
                        delta -= 1
                    if new_c == 4:
                        delta += 1
                cand_score = cur_score + delta
                if cand_score >= cur_score or rng.random() < math.exp((cand_score - cur_score) / T):
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        counts[k] = counts.get(k, 0) + 1
                        graph[i][j] = graph[j][i] = k
                    cur[i] = np_
                    cur_score = cand_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_pts = [list(p) for p in cur]
                    break
            T = max(T * cooling, T_min)

    return best_score, best_pts


def construct():
    start = time.time()
    deadline = start + 110.0
    rng = random.Random(12345)

    seeds = [
        _seed_grid(),
        _seed_staggered(),
        _seed_affine(),
        _seed_hexagonal(),
        _seed_lattice(),
        _seed_random(),
    ]

    global_best = -10**9
    global_pts = seeds[0]

    # Give each seed a share of the remaining budget.
    n_seeds = len(seeds)
    per = max(4.0, (deadline - time.time()) / n_seeds)

    for s in seeds:
        if time.time() >= deadline - 2.0:
            break
        seed_budget = min(time.time() + per, deadline - 2.0)
        try:
            sc, pts = _optimize(s, rng, seed_budget)
        except Exception:
            continue
        if sc > global_best:
            global_best = sc
            global_pts = pts

    # Final safety: ensure validity of the returned configuration.
    if global_pts is not None:
        g = _build_graph(global_pts, None)
        if not _validity(global_pts, g):
            # fall back to the plain grid (always valid, 6 exact-4 lines)
            global_pts = _seed_grid()

    return global_pts
