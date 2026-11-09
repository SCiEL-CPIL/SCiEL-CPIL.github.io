"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 20

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

# ---- Line bookkeeping: lines as frozensets of point indices, keyed by (dx,dy,offset) ----
def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(abs(dx), abs(dy))
    dx //= g; dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx; dy = -dy
    # offset = dy*x0 - dx*y0 (constant along the line)
    off = dy * p1[0] - dx * p1[1]
    return (dx, dy, off)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _build_lines(points):
    """Return dict line_key -> frozenset(indices) for all lines with >=2 points."""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            k = _line_key(points[i], points[j])
            if k in lines:
                lines[k] = lines[k] | {i, j}
            else:
                lines[k] = frozenset({i, j})
    return lines

def _has_four_collinear(points):
    lines = _build_lines(points)
    for fs in lines.values():
        if len(fs) >= 4:
            return True
    return False

# ---- Hypergraph representation for the 3-free subset (independent set) ----
def _build_hypergraph(points):
    """Return list of frozenset triples (collinear triples) among the 20 points."""
    lines = _build_lines(points)
    triples = []
    for fs in lines.values():
        idx = list(fs)
        for a, b, c in combinations(idx, 3):
            triples.append(frozenset((a, b, c)))
    # dedup (a triple can't appear in two different lines, but be safe)
    return list(set(triples))

def _build_conflict(triples):
    """conflict[i] = bitmask of points that form a collinear triple with i."""
    n = N
    conflict = [0] * n
    for t in triples:
        a, b, c = tuple(t)
        conflict[a] |= (1 << b) | (1 << c)
        conflict[b] |= (1 << a) | (1 << c)
        conflict[c] |= (1 << a) | (1 << b)
    return conflict

def _max_no_three_collinear(points):
    """Largest subset with no three collinear, via bitset branch-and-bound on the
    3-uniform hypergraph independent set (independent = no hyperedge fully inside)."""
    triples = _build_hypergraph(points)
    conflict = _build_conflict(triples)
    n = N
    full = (1 << n) - 1
    best = [0]
    # Precompute popcount via bin
    def popcount(x):
        return bin(x).count('1')

    # Greedy lower bound: take points in order, keep if no conflict with already taken
    greedy = 0
    taken = 0
    for i in range(n):
        bit = 1 << i
        if (conflict[i] & taken) == 0:
            taken |= bit
            greedy += 1
    best[0] = greedy

    # Order vertices by descending conflict degree for tighter pruning
    order = sorted(range(n), key=lambda i: -popcount(conflict[i]))
    # Map to a permutation so we process in that order; keep a bit array
    # We'll do recursive B&B over the ordered list using a "remaining" bitmask.

    import sys
    sys.setrecursionlimit(10000)

    # conflict in original index space; we operate on bitmasks directly.
    # Branching: at each step pick the highest set bit in 'avail' that is next in order.
    # Simpler: iterate in fixed order list; pass index position.
    # Use a stack-based recursion over position pos in order, with chosen mask.
    def bb(pos, chosen, avail):
        # upper bound
        if popcount(chosen) + popcount(avail) <= best[0]:
            return
        if pos == n:
            if popcount(chosen) > best[0]:
                best[0] = popcount(chosen)
            return
        i = order[pos]
        bit = 1 << i
        # Option 1: skip i
        bb(pos+1, chosen, avail & ~bit)
        # Option 2: take i if it doesn't conflict with any already-chosen point
        if (conflict[i] & chosen) == 0:
            # taking i removes from avail all points that conflict with i
            new_avail = avail & ~bit & ~(conflict[i])
            bb(pos+1, chosen | bit, new_avail)

    bb(0, 0, full)
    return best[0]

# ---- Local search: move a point onto an existing line to add collinear triples ----
def _candidate_on_line(points, idx, lines, used):
    """Yield integer points (x,y) within bounds, not used, that lie on one of the
    given lines (a line = list of >=2 existing indices) and add at least one new
    collinear triple without making a 4-collinear line."""
    # We just sample candidate coordinates on each line near the existing points.
    for line in lines:
        if len(line) < 2:
            continue
        if len(line) >= 3:
            # placing a 4th on this line -> 4 collinear, invalid
            continue
        # line has exactly 2 points: a, b
        a, b = tuple(line)
        pa, pb = points[a], points[b]
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        g = _gcd(abs(dx), abs(dy))
        dx //= g; dy //= g
        # integer points on the line: pa + t*(dx,dy)
        for t in range(-15, 16):
            if t == 0:
                continue
            x = pa[0] + t * dx
            y = pa[1] + t * dy
            if abs(x) > 100 or abs(y) > 100:
                continue
            if (x, y) in used:
                continue
            yield (x, y)

def _try_improve(points, deadline):
    """One pass: try to move each point to a spot that increases triple count
    (more packing) while keeping no-four-collinear. Returns new points or None."""
    used = set((p[0], p[1]) for p in points)
    lines = _build_lines(points)
    base_triples = len(_build_hypergraph(points))
    improved = False
    for idx in range(N):
        if time.time() > deadline:
            break
        used_no_idx = used - {(points[idx][0], points[idx][1])}
        best_gain = 0
        best_pos = None
        for (x, y) in _candidate_on_line(points, idx, lines, used_no_idx):
            new_points = points[:idx] + [[x, y]] + points[idx+1:]
            if _has_four_collinear(new_points):
                continue
            nt = len(_build_hypergraph(new_points))
            if nt > base_triples + best_gain:
                best_gain = nt - base_triples
                best_pos = (x, y)
        if best_pos is not None:
            points[idx] = [best_pos[0], best_pos[1]]
            improved = True
            # rebuild for subsequent idx
            used = set((p[0], p[1]) for p in points)
            lines = _build_lines(points)
            base_triples = len(_build_hypergraph(points))
    return points if improved else None

def _structured_initial():
    """A parabola-based structured start (few collinearities by design), then
    perturb to break any accidental 4-collinear."""
    pts = []
    for i in range(N):
        x = i - 10
        y = x * x - 50  # parabola y = x^2 - 50
        pts.append([x, y])
    # parabola has no 3 collinear? Actually a parabola y=x^2 has no 3 collinear.
    # Ensure within bounds and distinct.
    for p in pts:
        p[0] = max(-100, min(100, p[0]))
        p[1] = max(-100, min(100, p[1]))
    # Remove duplicates by shifting
    seen = set()
    out = []
    for p in pts:
        key = (p[0], p[1])
        if key in seen:
            p[1] = (p[1] + 1) % 201 - 100
            key = (p[0], p[1])
        seen.add(key)
        out.append(p)
    if len(out) != N or _has_four_collinear(out):
        # fall back to random
        out = _random_initial()
    return out

def _random_initial():
    while True:
        pts = []
        for _ in range(N):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in [(p[0], p[1]) for p in pts]:
                    pts.append([x, y])
                    break
        if not _has_four_collinear(pts):
            return pts

def construct():
    start = time.time()
    deadline = start + 110  # safety margin vs 150s kill

    # Start from a structured config, then pack via local search.
    points = _structured_initial()
    best_points = [p[:] for p in points]
    best_score = _max_no_three_collinear(points)

    # Iterated local search with line-targeted moves.
    restart = 0
    while time.time() < deadline - 5:
        # Do several improvement passes
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            res = _try_improve(points, deadline - 5)
            if res is None:
                break
            points = res
        score = _max_no_three_collinear(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
        # Random restart: perturb current best a little
        restart += 1
        points = [p[:] for p in best_points]
        # Perturb 3 random points to new random valid spots
        for _ in range(3):
            idx = random.randrange(N)
            for _ in range(200):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = points[:idx] + [[x, y]] + points[idx+1:]
                if (x, y) not in [(p[0], p[1]) for p in cand] and not _has_four_collinear(cand):
                    points = cand
                    break
        if time.time() > deadline - 20:
            break

    # Final: make sure best_points is valid
    if _has_four_collinear(best_points):
        best_points = _random_initial()
    return best_points
# EVOLVE-BLOCK-END
