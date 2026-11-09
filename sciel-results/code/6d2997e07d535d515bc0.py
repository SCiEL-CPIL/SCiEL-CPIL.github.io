"""Combine: line-segment compatibility (structural) + fast exact-4 scoring (refinement)."""
import time
import random
from math import gcd

_T0 = time.time()
_DEADLINE = 132.0  # safety margin before the 150 s wall-clock kill

# ---- 4-point line catalogue (structural representation) ----
# Each entry: (key, (x1,y1), (x2,y2), (x3,y3), (x4,y4), frozenset of 4 points)
_CATALOG = None
_CATALOG_BUILD_MS = None


def _line_key(p1, p2):
    """Canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a = dy
    b = -dx
    g = gcd(abs(a), abs(b))
    a //= g
    b //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)


def _count_exact4(points):
    """Return (num_exact_4_lines, has_5_collinear). Fast pair-count method."""
    n = len(points)
    pair_count = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k is None:
                continue
            pair_count[k] = pair_count.get(k, 0) + 1
    four = 0
    has5 = False
    for cnt in pair_count.values():
        if cnt == 6:
            four += 1
        elif cnt >= 10:
            has5 = True
    return four, has5


def _build_catalog():
    """Enumerate all 4-point lines over a bounded grid."""
    global _CATALOG
    if _CATALOG is not None:
        return
    N = 12
    pts = [(x, y) for x in range(-N, N + 1) for y in range(-N, N + 1)]
    m = len(pts)
    # pair -> list of points on that line
    line_points = {}
    for i in range(m):
        x1, y1 = pts[i]
        for j in range(i + 1, m):
            k = _line_key(pts[i], pts[j])
            if k is None:
                continue
            if k not in line_points:
                line_points[k] = [pts[i], pts[j]]
            else:
                line_points[k].append(pts[i])
                line_points[k].append(pts[j])
    cat = []
    for k, lst in line_points.items():
        s = set(lst)
        if len(s) == 4:
            quad = tuple(sorted(s))
            cat.append((k, quad, frozenset(quad)))
    _CATALOG = cat


def _greedy_select(lines):
    """Greedy maximal compatible subset. lines: list of (key, quad, fset)."""
    # Precompute pairwise shared-point counts for speed.
    L = len(lines)
    shared = [[0] * L for _ in range(L)]
    for i in range(L):
        fi = lines[i][2]
        for j in range(i + 1, L):
            s = len(fi & lines[j][2])
            shared[i][j] = s
            shared[j][i] = s
    selected = []
    for i in range(L):
        ok = True
        for j in selected:
            if shared[i][j] > 1:  # >1 shared => same line (4) or 5-collinear (>=2 with distinct)
                ok = False
                break
        if ok:
            selected.append(i)
    return selected


def _points_from_lines(lines, selected):
    out = []
    for i in selected:
        for p in lines[i][1]:
            if p not in out:
                out.append(p)
    return out


def _try_line_search():
    """Build catalogue, greedily select, return (points, score) or None."""
    _build_catalog()
    lines = _CATALOG
    if not lines:
        return None
    # Order lines by a heuristic: prefer lines whose points are 'central'/dense?
    # Simple: try a few orderings and keep the best union that is valid 16-pt.
    best = None
    # Deterministic first pass
    sel = _greedy_select(lines)
    pts = _points_from_lines(lines, sel)
    if len(set(map(tuple, pts))) == 16:
        sc, has5 = _count_exact4(pts)
        if not has5 and (best is None or sc > best[1]):
            best = (pts, sc)
    return best


def _perturb(points, rng):
    """Small random perturbation of a few points."""
    pts = [list(p) for p in points]
    for _ in range(rng.randint(2, 5)):
        i = rng.randrange(16)
        pts[i] = [pts[i][0] + rng.randint(-4, 4), pts[i][1] + rng.randint(-4, 4)]
    # dedupe
    seen = set()
    out = []
    for p in pts:
        if p[0] < -1000 or p[0] > 1000 or p[1] < -1000 or p[1] > 1000:
            return None
        t = (p[0], p[1])
        if t not in seen:
            seen.add(t)
            out.append([p[0], p[1]])
    if len(out) != 16:
        return None
    return out


def _local_improve(points, rng):
    """Local search: move single points to try to raise exact-4 count."""
    pts = [list(p) for p in points]
    cur, has5 = _count_exact4(pts)
    if has5:
        cur = -1
    best_pts = [list(p) for p in pts]
    best_score = cur
    for _ in range(400):
        if time.time() > _DEADLINE:
            break
        i = rng.randrange(16)
        ox, oy = pts[i]
        for _ in range(8):
            nx = ox + rng.randint(-6, 6)
            ny = oy + rng.randint(-6, 6)
            if nx < -1000 or nx > 1000 or ny < -1000 or ny > 1000:
                continue
            pts[i] = [nx, ny]
            # distinctness
            dup = False
            for j in range(16):
                if j != i and pts[j][0] == nx and pts[j][1] == ny:
                    dup = True
                    break
            if dup:
                continue
            sc, has5 = _count_exact4(pts)
            if has5:
                sc = -1
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in pts]
            else:
                pts[i] = [ox, oy]
    return best_pts, best_score


def construct():
    t_start = time.time()
    rng = random.Random(20240517)

    # Baseline: 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in base]
    best_score, _ = _count_exact4(best_pts)

    # Structural line-search phase
    try:
        res = _try_line_search()
        if res is not None:
            pts, sc = res
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in pts]
    except Exception:
        pass

    # If line search produced a strong config, refine it; otherwise refine baseline.
    if time.time() < _DEADLINE - 10:
        seed_pts = best_pts
        for attempt in range(4):
            if time.time() > _DEADLINE - 5:
                break
            if attempt == 0:
                start = seed_pts
            else:
                start = _perturb(seed_pts, rng)
                if start is None:
                    start = seed_pts
            cand, sc = _local_improve(start, rng)
            if sc > best_score:
                best_score = sc
                best_pts = [list(p) for p in cand]

    # Final validation; fall back to a valid 4x4 grid if anything went wrong.
    final, has5 = _count_exact4(best_pts)
    if has5 or final < 0 or len(set(map(tuple, best_pts))) != 16:
        final_pts = [[x, y] for x in range(4) for y in range(4)]
        return final_pts
    return best_pts


if __name__ == "__main__":
    r = construct()
    print(len(r), r[:4])
    print(_count_exact4(r))
