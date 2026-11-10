"""Rewritten program: grid seed + NumPy local search to maximize max doubled-area multiplicity."""
import time
import itertools
import numpy as np

_DEADLINE = 85.0  # safety margin under 90s kill

def _score(points):
    """Return the largest multiplicity of a single positive doubled area over all triples."""
    n = len(points)
    if n < 3:
        return 0
    P = np.asarray(points, dtype=np.int64)
    best = 0
    # iterate over pairs (i,j) as base, vectorize over k
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # area2 = | (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi) |
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            rest = P[j+1:]
            # cross = dx*(rest_y - yi) - dy*(rest_x - xi)
            cross = dx * (rest[:, 1] - P[i, 1]) - dy * (rest[:, 0] - P[i, 0])
            vals = np.abs(cross)
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            # count multiplicities
            vals, counts = np.unique(vals, return_counts=True)
            local_best = int(counts.max())
            if local_best > best:
                best = local_best
    return best

def _score_fast(points):
    """Full vectorized score over all triples (n=30 -> 4060 triples, cheap)."""
    n = len(points)
    P = np.asarray(points, dtype=np.int64)
    idx = np.arange(n)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing='ij')
    ii = ii[ii < jj].ravel()
    jj = jj[ii < jj].ravel()
    kk = kk[ii < jj].ravel()
    mask = jj < kk
    ii, jj, kk = ii[mask], jj[mask], kk[mask]
    A = P[ii]
    B = P[jj]
    C = P[kk]
    cross = (B[:,0]-A[:,0])*(C[:,1]-A[:,1]) - (B[:,1]-A[:,1])*(C[:,0]-A[:,0])
    vals = np.abs(cross)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    vals, counts = np.unique(vals, return_counts=True)
    return int(counts.max())

def construct():
    start = time.time()

    # ---- seed: 5x6 grid, centered, within [-30,30] ----
    def make_grid(rows, cols, ox=0, oy=0, sx=1, sy=1):
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = ox + c * sx
                y = oy + r * sy
                pts.append((x, y))
        return pts

    candidates = []
    # base 5x6
    candidates.append(make_grid(5, 6, ox=-3, oy=-2))
    # 6x5
    candidates.append(make_grid(6, 5, ox=-2, oy=-3))
    # a few shifted / scaled variants
    for sx, sy in [(1,1),(2,1),(1,2),(2,2)]:
        candidates.append(make_grid(5, 6, ox=-2, oy=-2, sx=sx, sy=sy))
        candidates.append(make_grid(6, 5, ox=-2, oy=-2, sx=sx, sy=sy))

    best_pts = None
    best_score = -1
    for c in candidates:
        # filter to distinct points in range
        c = [p for p in c if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
        if len(set(c)) != 30:
            continue
        s = _score_fast(c)
        if s > best_score:
            best_score = s
            best_pts = list(c)

    if best_pts is None:
        best_pts = make_grid(5, 6, ox=-3, oy=-2)
        best_score = _score_fast(best_pts)

    # ---- local search ----
    def neighborhood_move(pts, idx, dx, dy):
        x, y = pts[idx]
        nx, ny = x + dx, y + dy
        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
            return None
        new = list(pts)
        new[idx] = (nx, ny)
        if len(set(new)) != 30:
            return None
        return new

    # candidate deltas: small moves
    deltas = []
    for d in range(-2, 3):
        for e in range(-2, 3):
            if d == 0 and e == 0:
                continue
            deltas.append((d, e))

    improved = True
    while improved and (time.time() - start) < _DEADLINE - 3.0:
        improved = False
        # shuffle-ish: iterate points in a fixed but varied order
        order = list(range(30))
        for idx in order:
            if (time.time() - start) > _DEADLINE - 3.0:
                break
            for (dx, dy) in deltas:
                new = neighborhood_move(best_pts, idx, dx, dy)
                if new is None:
                    continue
                s = _score_fast(new)
                if s > best_score:
                    best_score = s
                    best_pts = new
                    improved = True
                    break  # restart ordering after an improvement

    # final validity guard
    assert len(best_pts) == 30
    assert len(set(best_pts)) == 30
    assert all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in best_pts)
    return list(best_pts)
