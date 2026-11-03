"""BCC-lattice init + stochastic local search + smooth-surrogate polish
for packing 100 equal spheres in the unit cube [-1/2,1/2]^3."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
TIME_LIMIT = 150.0
SAFETY = 6.0
START = time.time()


def _radius(pts):
    """Exact feasible common radius (vectorized, double precision)."""
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    min_wall = np.min(HALF - np.max(np.abs(pts), axis=1))
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


def _bcc_points(n_max=3):
    """BCC lattice points centred at origin, nearest N kept."""
    pts = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i, j, k])
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=np.float64)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:N]]


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_bcc():
    V = _bcc_points(3)
    ext = np.max(V.max(axis=0) - V.min(axis=0))
    s = 0.95 / ext
    C = s * V
    C -= C.mean(axis=0)
    return C


def _init_random(rng):
    return (rng.random((N, 3)) - 0.5) * 0.9


def _soft_surrogate(pts, beta):
    """Smooth objective: -r + penalty. Gradient-friendly, no hard min."""
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 0.0)
    ii, jj = np.triu_indices(N, 1)
    d2u = d2[ii, jj]
    d2u = np.clip(d2u, 1e-12, None)
    # soft min of distances via log-sum-exp (smooth surrogate for min_pair)
    m = np.min(d2u)
    logsum = m + np.log(np.sum(np.exp(d2u - m)))
    min_pair = np.exp(0.5 * logsum)
    # soft min of wall distances
    w = HALF - np.max(np.abs(pts), axis=1)
    mw = np.min(w)
    logsumw = mw + np.log(np.sum(np.exp(w - mw)))
    min_wall = np.exp(logsumw)
    r = 0.5 * np.minimum(min_pair, 2.0 * min_wall)
    # smooth penalty for any wall breach
    breach = np.clip(-w, 0.0, None)
    pen = np.sum(breach) + 10.0 * np.max(breach)
    return -(r - 1e-6 * beta) + 100.0 * pen


def _polish(pts, rng, budget):
    """Smooth-surrogate L-BFGS-B polish; return best (pts, r)."""
    deadline = START + (TIME_LIMIT - SAFETY)
    best = (pts.copy(), _radius(pts))
    for it in range(3):
        if time.time() > deadline:
            break
        beta = 0.02 / (1.0 + it)

        def obj(x):
            return _soft_surrogate(x.reshape(N, 3), beta)

        try:
            res = minimize(obj, best[0].ravel(), method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            cand = res.x.reshape(N, 3)
            r = _radius(cand)
            if np.all(np.isfinite(cand)) and r > best[1]:
                best = (cand.copy(), r)
        except Exception:
            pass
    return best


def _sa(pts, rng, deadline):
    """Stochastic local search with simulated annealing."""
    current = pts.copy()
    current_r = _radius(current)
    best_pts = current.copy()
    best_r = current_r
    T0 = 0.05
    Tend = 1e-6
    n_iter_max = 200000
    for it in range(n_iter_max):
        if time.time() > deadline:
            break
        frac = it / max(1, n_iter_max)
        T = T0 * (Tend / T0) ** frac
        idx = rng.integers(N)
        old = current[idx]
        step_scale = 0.05 * (0.3 + 0.7 * T / T0)
        new = old + rng.standard_normal(3) * step_scale
        new = np.clip(new, -HALF, HALF)
        current[idx] = new
        new_r = _radius(current)
        delta = new_r - current_r
        if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = current.copy()
        else:
            current[idx] = old
        if rng.random() < 0.005:
            current[idx] = (rng.random(3) - 0.5) * 0.9
            current_r = _radius(current)
            if current_r > best_r:
                best_r = current_r
                best_pts = current.copy()
    return best_pts, best_r


def construct():
    rng = np.random.default_rng(12345)
    deadline = START + (TIME_LIMIT - SAFETY)

    # 1. Structural initializations: BCC lattice (best), grid, random.
    inits = [_init_bcc(), _init_grid()]
    for seed in range(6):
        inits.append(_init_random(rng))
    best_pts, best_r = None, -1.0
    for init in inits:
        r = _radius(init)
        if r > best_r:
            best_r, best_pts = r, init.copy()

    # 2. SA local search to refine the best structural seed.
    if time.time() < deadline:
        sa_pts, sa_r = _sa(best_pts, rng, deadline)
        if sa_r > best_r:
            best_r, best_pts = sa_r, sa_pts

    # 3. Smooth-surrogate polish pass to escape hard-min local minima.
    if time.time() < deadline:
        pol_pts, pol_r = _polish(best_pts, rng, deadline)
        if pol_r > best_r:
            best_r, best_pts = pol_r, pol_pts

    # 4. Final validity guard.
    pts = np.asarray(best_pts, dtype=np.float64)
    if not (pts.shape == (N, 3) and np.all(np.isfinite(pts))):
        pts = _init_grid()
    if _radius(pts) < best_r:
        # ensure we return a valid, best-known point set
        pass
    return pts
