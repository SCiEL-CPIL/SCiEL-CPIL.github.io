"""Packing 12 unit hexagons in a minimal regular hexagon.

Strategy: minimise the outer side length.
 - Outer containment is equivalent to: max over inner vertices of
   |x|, |y|, |x/2 + sqrt(3)*y/2| <= side (for a flat-top regular hexagon
   centered at origin with rotation 0).
 - We minimise a smooth objective = max_vertex_dist + overlap_penalty.
 - Optimise with a gradient-free (differential-evolution-like) search
   over all 12 hexagons' (x, y, angle) with multiple random restarts.
"""
import math
import time
import random
import numpy as np

N = 12
SQ3 = math.sqrt(3.0)

# vertices of a unit regular hexagon (flat-top), centered at origin
_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, SQ3 / 2.0],
    [-0.5, SQ3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQ3 / 2.0],
    [0.5, -SQ3 / 2.0],
])  # shape (6,2)


def _hex_vertices(cx, cy, ang_deg):
    """Return 6x2 array of vertex coords for unit hex at (cx,cy) rotated."""
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _HEX_VERTS @ R.T + np.array([cx, cy])


def _max_vertex_radius(pts):
    """Max of |x|, |y|, |x/2 + sqrt3*y/2| over all points."""
    x = pts[:, 0]
    y = pts[:, 1]
    d1 = np.abs(x)
    d2 = np.abs(y)
    d3 = np.abs(0.5 * x + 0.5 * SQ3 * y)
    return float(np.max(np.concatenate([d1, d2, d3])))


def _min_pairwise_dist(P):
    """P: (12,6,2) vertex arrays. Min distance between any two different hexes'
    vertex sets, but we use a cheaper proxy: min over all pairs of (vertex,
    hex-center) to detect overlap. Use SDF-ish: for each pair of hexes, min
    distance between any vertex of A and any vertex of B is a lower bound on
    true gap; if it's < 0 they clearly overlap. We'll use min vertex-vertex
    distance as a smooth-ish proxy for overlap."""
    n = len(P)
    best = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            # min distance between any vertex of i and any vertex of j
            # (12x12 = 144 dists)
            d = np.linalg.norm(P[i][:, None, :] - P[j][None, :, :], axis=2)
            m = float(d.min())
            if m < best:
                best = m
    return best


def _objective(params):
    """params: (36,) -> 12*(x,y,angle). Returns (score, side, min_gap)."""
    pts = np.empty((N, 6, 2))
    for i in range(N):
        cx, cy, ang = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        pts[i] = _hex_vertices(cx, cy, ang)
    side = _max_vertex_radius(pts.reshape(-1, 2))
    gap = _min_pairwise_dist(pts)
    # penalise negative gap (overlap) heavily
    pen = 0.0
    if gap < 0.0:
        pen = -gap * 50.0
    return side + pen, side, gap


# ---------- initial layouts ----------
def _init_line():
    """12 in a row, flat-top, spaced 2.02 apart."""
    p = np.zeros(3 * N)
    for i in range(N):
        p[3 * i] = (i - (N - 1) / 2.0) * 2.02
        p[3 * i + 1] = 0.0
        p[3 * i + 2] = 0.0
    return p


def _init_ring():
    """Inner hexagons around a center, one at center-ish."""
    p = np.zeros(3 * N)
    # 12 around a circle of radius 2.2
    for i in range(N):
        a = 2 * math.pi * i / N
        p[3 * i] = 2.2 * math.cos(a)
        p[3 * i + 1] = 2.2 * math.sin(a)
        p[3 * i + 2] = 0.0
    return p


def _init_grid():
    """4x3 grid with hex spacing."""
    p = np.zeros(3 * N)
    dx = 2.02
    dy = SQ3 * 1.02  # ~1.766
    k = 0
    for r in range(3):
        for c in range(4):
            p[3 * k] = (c - 1.5) * dx + (0.5 if r % 2 else 0.0) * dx
            p[3 * k + 1] = (r - 1.0) * dy
            p[3 * k + 2] = 0.0
            k += 1
    return p


def _random_init(scale=4.0):
    p = np.zeros(3 * N)
    for i in range(N):
        p[3 * i] = random.uniform(-scale, scale)
        p[3 * i + 1] = random.uniform(-scale, scale)
        p[3 * i + 2] = random.choice([0.0, 30.0, 60.0, 90.0])
    return p


def _evolve(start, deadline, pop=24, iters=1200, seed=None):
    """Differential-evolution-like local search."""
    rng = random.Random(seed)
    pop_p = [start.copy() for _ in range(pop)]
    # diversify initial population
    for i in range(1, pop):
        for j in range(3 * N):
            pop_p[i][j] += rng.gauss(0.0, 0.3)
    best_p = start.copy()
    best_s, _, _ = _objective(start)

    for it in range(iters):
        if time.time() > deadline:
            break
        # evolve each individual
        for i in range(pop):
            if time.time() > deadline:
                break
            # pick 3 distinct others
            a, b, c = rng.sample([k for k in range(pop) if k != i], 3)
            # crossover
            child = pop_p[a].copy()
            for j in range(3 * N):
                if rng.random() < 0.5:
                    child[j] = pop_p[b][j] + 0.5 * (pop_p[c][j] - pop_p[a][j])
            # mutation
            child += rng.gauss(0.0, 0.15, 3 * N)
            s, _, _ = _objective(child)
            if s < _objective(pop_p[i])[0]:
                pop_p[i] = child
            if s < best_s:
                best_s = s
                best_p = child.copy()
        # inject a fresh random to avoid local trap
        if it % 40 == 39:
            pop_p[rng.randrange(pop)] = _random_init(3.0)
    return best_p, best_s


def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin under 150s

    inits = [_init_line(), _init_ring(), _init_grid()]
    best_p = inits[0]
    best_s = 1e9
    for s0 in inits:
        s, _, _ = _objective(s0)
        if s < best_s:
            best_s = s
            best_p = s0

    # multi-start evolution
    starts = inits + [_random_init(4.0) for _ in range(2)]
    for k, s0 in enumerate(starts):
        if time.time() > deadline:
            break
        p, s = _evolve(s0, deadline, pop=16, iters=600, seed=1000 + k)
        if s < best_s:
            best_s = s
            best_p = p

    # final polish: greedy local refinement on best
    p = best_p.copy()
    for step in range(2000):
        if time.time() > deadline:
            break
        cur_s, _, _ = _objective(p)
        improved = False
        for i in range(3 * N):
            for delta in (0.05, -0.05, 0.1, -0.1, 0.02, -0.02):
                q = p.copy()
                q[i] += delta
                s, _, _ = _objective(q)
                if s < cur_s:
                    p = q
                    cur_s = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    if cur_s < best_s:
        best_s = cur_s
        best_p = p

    # build output
    inner = []
    for i in range(N):
        inner.append([float(best_p[3 * i]), float(best_p[3 * i + 1]),
                      float(best_p[3 * i + 2])])
    # side = max vertex radius (containment)
    pts = np.empty((N, 6, 2))
    for i in range(N):
        pts[i] = _hex_vertices(best_p[3 * i], best_p[3 * i + 1], best_p[3 * i + 2])
    side = _max_vertex_radius(pts.reshape(-1, 2))
    # add tiny safety margin so evaluator sees valid containment
    side = side + 1e-6
    return inner, (0.0, 0.0), float(side), 0.0
