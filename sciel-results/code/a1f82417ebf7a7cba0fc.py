"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Structural rewrite: global simulated annealing + targeted 'lift' moves that attack the
smallest triangle directly, plus a local polish. This replaces the previous
gradient/Newton/ring-symmetry family with a non-gradient global search that is robust
to the non-differentiable min-triangle objective.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 13

_DEADLINE = time.time() + 145.0  # wall-clock guard (150 s kill, 5 s safety margin)


def _time_ok(margin=0.0):
    return time.time() < _DEADLINE - margin


def _all_areas(pts):
    """Vectorized triangle areas for all C(n,3) triples, stored in a flat array.

    Order: idx = (i * (2n - i - 3) // 2) + (j - i - 1) + (k - j - 1)
    """
    n = len(pts)
    m = n * (n - 1) * (n - 2) // 6
    areas = np.empty(m, dtype=np.float64)
    p = 0
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        mlen = len(rest)
        for j in range(mlen - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            cross = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            a = 0.5 * np.abs(cross)
            cnt = a.size
            areas[p:p + cnt] = a
            p += cnt
    return areas


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    areas = _all_areas(pts)
    if areas.size == 0:
        return 0.0
    return float(np.min(areas)) / hull_a


def _smallest_triangle(pts):
    """Return (min_area, (i, j, k)) for the smallest triangle."""
    n = len(pts)
    best = float('inf')
    bi = bj = bk = 0
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        mlen = len(rest)
        for j in range(mlen - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            cross = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            a = 0.5 * np.abs(cross)
            li = int(np.argmin(a))
            va = a[li]
            if va < best:
                best = va
                bi, bj, bk = i, i + 1 + j, i + 2 + j + li
    return best, (bi, bj, bk)


def _lift(pts, i, j, k, strength=0.02):
    """Move point i directly away from the line through j and k (increases area of ijk)."""
    pts = pts.copy()
    pj, pk, pi = pts[j], pts[k], pts[i]
    d = pk - pj
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        # degenerate: push i away from j
        d = pi - pj
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            return pts
        n = d / nrm
        # perpendicular
        perp = np.array([-n[1], n[0]])
        # choose side away from current position relative to line
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    else:
        n = d / nrm
        perp = np.array([-n[1], n[0]])
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    return pts


def _local_polish(pts, max_iter=3000):
    """Deterministic local search: repeatedly lift the smallest triangle's points."""
    pts = pts.copy()
    cur = _score(pts)
    for _ in range(max_iter):
        if not _time_ok(2.0):
            break
        _, (i, j, k) = _smallest_triangle(pts)
        improved = False
        for pt in (i, j, k):
            for s in (0.005, 0.01, 0.02, 0.04, 0.08):
                cand = _lift(pts, pt, j if pt != j else i, k if pt != k else i, strength=s)
                # ensure we lift pt away from the other two
                if pt == i:
                    cand = _lift(pts, i, j, k, strength=s)
                elif pt == j:
                    cand = _lift(pts, j, i, k, strength=s)
                else:
                    cand = _lift(pts, k, i, j, strength=s)
                sc = _score(cand)
                if sc > cur + 1e-12:
                    pts, cur, improved = cand, sc, True
                    break
            if improved:
                break
        if not improved:
            break
    return pts, cur


def _simulated_anneal(pts, rng, t0=0.02, t1=1e-5, n_steps=30000, lift_frac=0.15):
    """Global SA with random jitter + targeted lift moves."""
    pts = pts.copy()
    cur = _score(pts)
    best = cur
    best_pts = pts.copy()
    n = len(pts)
    for step in range(n_steps):
        if not _time_ok(3.0):
            break
        frac = step / max(1, n_steps)
        T = t0 * (t1 / t0) ** frac
        if rng.rand() < lift_frac:
            # targeted lift on the smallest triangle
            _, (i, j, k) = _smallest_triangle(pts)
            pt = int(rng.randint(3))
            a, b, c = (i, j, k)
            if pt == 0:
                cand = _lift(pts, a, b, c, strength=rng.uniform(0.005, 0.05))
            elif pt == 1:
                cand = _lift(pts, b, a, c, strength=rng.uniform(0.005, 0.05))
            else:
                cand = _lift(pts, c, a, b, strength=rng.uniform(0.005, 0.05))
        else:
            # random jitter on one or two points
            cand = pts.copy()
            npts = 1 if rng.rand() < 0.7 else 2
            for _ in range(npts):
                idx = int(rng.randint(n))
                cand[idx] += rng.normal(0, T, size=2)
        sc = _score(cand)
        if sc >= cur or rng.rand() < np.exp((sc - cur) / max(T, 1e-12)):
            cur, pts = sc, cand
            if cur > best:
                best, best_pts = cur, cand.copy()
    return best_pts, best


def construct():
    rng = np.random.RandomState(12345)

    # ---- Diverse initializations ----
    inits = []
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles13), np.sin(angles13)]))
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    inits.append(np.vstack([np.column_stack([np.cos(angles12), np.sin(angles12)]), [[0.0, 0.0]]]))

    # concentric rings
    ring_structures = [
        [13], [1, 12], [1, 6, 6], [7, 6], [2, 11], [3, 10], [4, 9], [5, 8], [6, 7],
        [1, 3, 9], [1, 4, 8], [1, 5, 7], [2, 5, 6], [1, 2, 3, 7],
    ]
    for rc in ring_structures:
        nr = len(rc)
        base_r = [1.0 / (i + 1) for i in range(nr)]
        for ph in [0.0, np.pi / 6.0, np.pi / 3.0]:
            pts = []
            for i, count in enumerate(rc):
                if count == 0:
                    continue
                ang = ph + 2 * np.pi * np.arange(count) / count
                pts.append(base_r[i] * np.column_stack([np.cos(ang), np.sin(ang)]))
            if pts:
                inits.append(np.vstack(pts))

    # random in unit disk
    for _ in range(6):
        r = np.sqrt(rng.rand(N))
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # jittered regular 13-gon
    for jit in [0.02, 0.05, 0.1]:
        rr = 1.0 + jit * rng.randn(N)
        inits.append(np.column_stack([rr * np.cos(angles13), rr * np.sin(angles13)]))

    # ---- Score initializations, keep top distinct candidates ----
    scored = []
    for ip in inits:
        if not _time_ok(40.0):
            break
        scored.append((_score(ip), ip))
    scored.sort(key=lambda x: -x[0])

    top = []
    for s, p in scored:
        if not any(np.allclose(p, q, atol=1e-5) for q in top):
            top.append(p)
        if len(top) >= 8:
            break

    best_score = scored[0][0]
    best_pts = scored[0][1].copy()

    # ---- Phase 1: local polish on each top candidate ----
    for p in top:
        if not _time_ok(10.0):
            break
        try:
            lp, ls = _local_polish(p)
            if ls > best_score:
                best_score, best_pts = ls, lp.copy()
        except Exception:
            pass

    # ---- Phase 2: simulated annealing from best + a few others ----
    for p in top[:4]:
        if not _time_ok(8.0):
            break
        try:
            ap, ascr = _simulated_anneal(p, rng, t0=0.02, t1=1e-5, n_steps=20000)
            if ascr > best_score:
                best_score, best_pts = ascr, ap.copy()
        except Exception:
            pass
        # polish the SA result
        if _time_ok(3.0):
            try:
                lp, ls = _local_polish(ap)
                if ls > best_score:
                    best_score, best_pts = ls, lp.copy()
            except Exception:
                pass

    # ---- Phase 3: SA + polish on the global best ----
    if _time_ok(6.0):
        try:
            ap, ascr = _simulated_anneal(best_pts, rng, t0=0.01, t1=1e-6, n_steps=25000)
            if ascr > best_score:
                best_score, best_pts = ascr, ap.copy()
        except Exception:
            pass
    if _time_ok(4.0):
        try:
            lp, ls = _local_polish(best_pts, max_iter=4000)
            if ls > best_score:
                best_score, best_pts = ls, lp.copy()
        except Exception:
            pass

    return best_pts
