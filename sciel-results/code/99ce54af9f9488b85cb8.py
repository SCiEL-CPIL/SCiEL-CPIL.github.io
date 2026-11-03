import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13
N_HULL = 7
N_IN = 6
TIME_LIMIT = 120.0
DEADLINE = time.time() + TIME_LIMIT - 10.0  # safety margin

# precompute all triangle index triples
_TRI = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
_TRI = np.array(_TRI, dtype=np.int64)


def _tri_areas(pts):
    a = pts[_TRI[:, 0]]
    b = pts[_TRI[:, 1]]
    c = pts[_TRI[:, 2]]
    return 0.5 * np.abs(np.cross(b - a, c - a))


def _hull_area(pts):
    if np.any(np.diff(np.sort(np.unique(pts[:, 0]))) == 0) and np.any(
        np.diff(np.sort(np.unique(pts[:, 1]))) == 0
    ):
        return 0.0
    try:
        h = ConvexHull(pts)
        return h.volume  # 2-D: this is area
    except Exception:
        return 0.0


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)


def _softmin(areas, tau):
    # negative log-sum-exp gives a smooth approximation of -min; we want min
    m = np.min(areas)
    return m - tau * np.log(np.mean(np.exp(-(areas - m) / tau)))


def _concentric_init(rng):
    # 7 outer hull vertices on a near-regular heptagon (slightly perturbed)
    outer = []
    for i in range(N_HULL):
        ang = 2 * np.pi * i / N_HULL + rng.uniform(-0.12, 0.12)
        r = 1.0 + rng.uniform(-0.05, 0.05)
        outer.append([r * np.cos(ang), r * np.sin(ang)])
    # 6 inner points on two concentric rings
    inner = []
    for ring, (rr, cnt, ph) in enumerate([(0.55, 3, 0.0), (0.28, 3, np.pi / 3)]):
        for i in range(cnt):
            ang = ph + 2 * np.pi * i / cnt + rng.uniform(-0.15, 0.15)
            r = rr + rng.uniform(-0.04, 0.04)
            inner.append([r * np.cos(ang), r * np.sin(ang)])
    pts = np.array(outer + inner, dtype=float)
    return pts


def _improve(pts, rng, budget):
    best = pts.copy()
    best_score = _score(best)
    n = len(pts)
    # anneal softmin temperature
    tau0, tau1 = 0.05, 0.0005
    iters = 600
    for it in range(iters):
        if time.time() > budget:
            break
        frac = it / max(1, iters - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        P = best.copy()
        areas = _tri_areas(P)
        ha = _hull_area(P)
        if ha <= 1e-12:
            break
        # objective = softmin(areas)/ha  (we maximize)
        # gradient of softmin w.r.t. each point via finite differences is expensive;
        # use coordinate perturbation on worst triangles instead.
        obj = _softmin(areas, tau) / ha
        # find worst (smallest) triangles
        order = np.argsort(areas)
        worst = _TRI[order[:20]]
        improved = False
        step = 0.02 * (1.0 - 0.7 * frac)
        for (i, j, k) in worst:
            for idx in (i, j, k):
                base = P[idx].copy()
                for dx in (step, -step, 0.0):
                    for dy in (step, -step, 0.0):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        P[idx] = base + np.array([dx, dy])
                        a2 = _tri_areas(P)
                        ha2 = _hull_area(P)
                        if ha2 <= 1e-12:
                            P[idx] = base
                            continue
                        o2 = _softmin(a2, tau) / ha2
                        if o2 > obj + 1e-12:
                            obj = o2
                            improved = True
                        else:
                            P[idx] = base
        if improved:
            s = _score(P)
            if s > best_score:
                best = P.copy()
                best_score = s
    return best, best_score


def construct():
    rng = np.random.default_rng(13)
    best = None
    best_score = -1.0
    # a few diverse initializations, each refined
    for seed in range(6):
        if time.time() > DEADLINE:
            break
        rng = np.random.default_rng(13 + seed * 7919)
        init = _concentric_init(rng)
        # small random jitter for diversity
        init = init + rng.normal(0, 0.02, init.shape)
        cand, sc = _improve(init, rng, DEADLINE)
        if sc > best_score:
            best_score = sc
            best = cand
    # final exact local polish on the true (nonsmooth) min objective
    if best is not None and time.time() < DEADLINE:
        P = best.copy()
        cur = _score(P)
        for it in range(2000):
            if time.time() > DEADLINE:
                break
            areas = _tri_areas(P)
            ha = _hull_area(P)
            if ha <= 1e-12:
                break
            cur = float(np.min(areas) / ha)
            order = np.argsort(areas)
            worst = _TRI[order[:15]]
            step = 0.004
            moved = False
            for (i, j, k) in worst:
                for idx in (i, j, k):
                    base = P[idx].copy()
                    for dx in (step, -step):
                        for dy in (step, -step):
                            P[idx] = base + np.array([dx, dy])
                            a2 = _tri_areas(P)
                            ha2 = _hull_area(P)
                            if ha2 <= 1e-12:
                                P[idx] = base
                                continue
                            s2 = float(np.min(a2) / ha2)
                            if s2 > cur + 1e-12:
                                cur = s2
                                moved = True
                            else:
                                P[idx] = base
            if not moved:
                step *= 0.6
                if step < 1e-5:
                    break
        if cur > best_score:
            best_score = cur
            best = P
    if best is None:
        best = _concentric_init(np.random.default_rng(13))
    return np.asarray(best, dtype=float)
