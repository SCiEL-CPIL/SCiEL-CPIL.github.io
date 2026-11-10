"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 110.0  # safety margin before 120 s evaluator kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _min_radius(pts):
    """Return the achievable common radius for a set of centres (double precision)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # container clearance
    d_center = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - d_center.max()
    # pairwise half-distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d.min()
    return min(r_cont, r_pair)


def _score(pts):
    return _min_radius(pts)


def _hex_init(n=N):
    """Hexagonal lattice packed into unit disk, take n nearest to origin."""
    # choose spacing so we get >= n points
    for spacing in [0.06, 0.058, 0.056, 0.055, 0.054, 0.053, 0.052, 0.051, 0.050, 0.048, 0.046, 0.044, 0.042, 0.040]:
        pts = []
        R = 1.0
        dy = spacing * np.sqrt(3.0) / 2.0
        j = 0
        while True:
            y = j * dy
            if y > R:
                break
            for i in range(-20, 21):
                x = (i + 0.5 * (j % 2)) * spacing
                if x * x + y * y <= (R - spacing / 2.0) ** 2:
                    pts.append((x, y))
            j += 1
        arr = np.array(pts)
        if len(arr) >= n:
            r = np.hypot(arr[:, 0], arr[:, 1])
            order = np.argsort(r, kind="stable")
            return arr[order[:n]].copy()
    # fallback random
    return _random_init()


def _spiral_init(n=N):
    """Sunflower / Vogel spiral filling the disk."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    ang = np.arange(n) * golden * 2.0 * np.pi
    # radius grows with sqrt to fill area uniformly
    r = np.sqrt(np.arange(n) / float(n)) * 0.999
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def _rings_init(n=N):
    """Concentric rings with jitter-free placement."""
    pts = [(0.0, 0.0)]
    k = 1
    while len(pts) < n:
        # circumference 2*pi*k*spacing ~ place ~ 2*pi*k*spacing/spacing = 2*pi*k points
        m = int(np.ceil(2.0 * np.pi * k * 0.5))  # heuristic spacing ~0.5 in ring units
        m = max(m, 6)
        ang = np.linspace(0, 2.0 * np.pi, m, endpoint=False)
        rr = 0.999 * k / 5.0
        for a in ang:
            if len(pts) < n:
                pts.append((rr * np.cos(a), rr * np.sin(a)))
        k += 1
    return np.array(pts[:n])


def _random_init(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(N, 2))
    keep = np.hypot(pts[:, 0], pts[:, 1]) <= 0.999
    while keep.sum() < N:
        extra = rng.uniform(-1, 1, size=(N - keep.sum(), 2))
        pts = np.vstack([pts[keep], extra])
        keep = np.hypot(pts[:, 0], pts[:, 1]) <= 0.999
    return pts[:N]


def _relax(pts, seed, max_iter, dt_base=0.02, temp_base=0.5):
    """Repulsion + confinement relaxation. Returns (best_pts, best_score)."""
    rng = np.random.default_rng(seed)
    pts = np.array(pts, dtype=float, copy=True)
    best = _score(pts)
    best_pts = pts.copy()

    # target radius estimate
    r_target = 0.052
    n = N

    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        # pairwise differences
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)

        # repulsive force: push apart when closer than 2*r_target
        # F = (1 - d/(2r)) * direction, only when d < 2r
        thresh = 2.0 * r_target
        mask = d < thresh
        strength = np.where(mask, (1.0 - d / thresh) ** 2, 0.0)
        # direction unit vectors
        d_safe = np.where(d > 1e-12, d, 1.0)
        dirx = diff[:, :, 0] / d_safe
        diry = diff[:, :, 1] / d_safe
        fx = (strength * dirx).sum(axis=1)
        fy = (strength * diry).sum(axis=1)

        # confinement: push inward if outside (1 - r_target)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        Rlim = 1.0 - r_target
        conf = np.where(rad > Rlim, (rad - Rlim) / r_target, 0.0)
        # inward direction = -pos/rad
        rad_safe = np.where(rad > 1e-12, rad, 1.0)
        fx -= conf * pts[:, 0] / rad_safe
        fy -= conf * pts[:, 1] / rad_safe

        # small random jitter to escape local minima
        jitter = rng.normal(0, temp_base * dt_base, size=(n, 2))
        fx += jitter[:, 0]
        fy += jitter[:, 1]

        # limit step size
        fmag = np.sqrt(fx * fx + fy * fy)
        fmag_safe = np.where(fmag > 1e-12, fmag, 1.0)
        step = dt_base
        fx = fx / fmag_safe * np.minimum(fmag, 1.0) * step
        fy = fy / fmag_safe * np.minimum(fmag, 1.0) * step

        pts[:, 0] += fx
        pts[:, 1] += fy

        # hard clamp to disk
        rad = np.hypot(pts[:, 0], pts[:, 1])
        over = rad > Rlim
        if over.any():
            scale = Rlim / rad[over]
            pts[over, 0] *= scale
            pts[over, 1] *= scale

        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()

        # anneal
        dt_base *= 0.995
        temp_base *= 0.998

    return best_pts, best


def _local_optimize(pts, seed, max_iter=200):
    """Greedy coordinate descent on the minimum constraint (both pair and container)."""
    pts = np.array(pts, dtype=float, copy=True)
    best = _score(pts)
    best_pts = pts.copy()
    rng = np.random.default_rng(seed)

    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        # find the most violated constraint
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        r_pair = 0.5 * d[i, j]
        rad = np.hypot(pts[:, 0], pts[:, 1])
        k = int(np.argmax(rad))
        r_cont = 1.0 - rad[k]
        if r_pair < r_cont:
            # move i and j apart along their connecting line
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            L = np.hypot(dx, dy)
            if L < 1e-12:
                pts[i, 0] += 1e-4
                pts[j, 0] -= 1e-4
            else:
                ux, uy = dx / L, dy / L
                eps = 1e-4
                pts[i, 0] += ux * eps
                pts[i, 1] += uy * eps
                pts[j, 0] -= ux * eps
                pts[j, 1] -= uy * eps
        else:
            # pull k inward
            ux, uy = pts[k, 0], pts[k, 1]
            L = np.hypot(ux, uy)
            if L < 1e-12:
                continue
            eps = 1e-4
            pts[k, 0] -= ux / L * eps
            pts[k, 1] -= uy / L * eps

        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()
    return best_pts, best


def construct():
    global _T0
    _T0 = time.time()
    _DEADLINE = 110.0

    best_pts = None
    best_score = -1.0

    # Build a pool of diverse initial configurations
    inits = []
    inits.append(("hex", _hex_init()))
    inits.append(("spiral", _spiral_init()))
    inits.append(("rings", _rings_init()))
    # a few random starts
    for s in (1, 2, 3):
        inits.append((f"rand{s}", _random_init(s)))

    # Phase 1: relax each with a moderate budget, keep improving
    for idx, (name, init) in enumerate(inits):
        if _remaining() < 15.0:
            break
        # give earlier starts more iterations
        max_iter = 600 if idx < 3 else 400
        pts, s = _relax(init, seed=100 + idx, max_iter=max_iter)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 2: local optimization on the best, plus a couple of random restarts if time
    if _remaining() > 20.0 and best_pts is not None:
        pts, s = _local_optimize(best_pts, seed=7, max_iter=300)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 3: if still time, try a couple more random relaxations from scratch
    if _remaining() > 30.0:
        for s in (11, 12):
            if _remaining() < 15.0:
                break
            init = _random_init(s)
            pts, sc = _relax(init, seed=200 + s, max_iter=400)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
            # local polish
            if _remaining() > 10.0:
                pts2, sc2 = _local_optimize(pts, seed=300 + s, max_iter=150)
                if sc2 > best_score:
                    best_score = sc2
                    best_pts = pts2.copy()

    # Final safety: ensure a valid configuration is returned
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init()

    # Final local polish with whatever time remains
    if _remaining() > 8.0:
        pts, s = _local_optimize(best_pts, seed=999, max_iter=200)
        if s > best_score:
            best_pts = pts.copy()
            best_score = s

    # Clamp to be strictly inside
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    over = rad > 0.999999
    if over.any():
        scale = 0.999999 / rad[over]
        best_pts[over, 0] *= scale
        best_pts[over, 1] *= scale

    return best_pts[:N]
# EVOLVE-BLOCK-END
