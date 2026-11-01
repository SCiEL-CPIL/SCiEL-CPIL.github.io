"""Optimize 200 equal circles in unit square via smooth soft-min + L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 150.0
MARGIN = 12.0  # safety margin seconds before the hard kill


def _score(pts):
    """Largest common radius for a set of centres (raw value)."""
    if pts is None or len(pts) != N or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # wall distance (distance from centre to nearest side of [-1/2,1/2]^2)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    wall_min = float(np.min(wall))
    if wall_min <= 0:
        return 0.0
    # pairwise distances
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_min = float(np.min(d)) / 2.0
    return min(wall_min, pair_min)


def _softmin_objective(pts, lam):
    """
    Smooth surrogate of -r. We want to maximize r = min over a set S of
    (pairwise/2) and (wall). Use soft-min:
        r_soft = -(1/lam) * log( sum_j exp(-lam * s_j) )
    where s_j are the "distance" terms (pairwise/2 and wall distances).
    Maximize r_soft  <=>  minimize f = -r_soft = (1/lam) log(sum exp(-lam s_j)).
    This is smooth and differentiable.
    """
    pts = np.asarray(pts)
    # wall distances
    wall = 0.5 - np.max(np.abs(pts), axis=1)  # (N,)
    # pairwise /2
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    # gather all terms
    terms = np.concatenate([wall, pair.ravel()])
    # soft-min with log-sum-exp stability
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    z = np.exp(-lam * (terms - m))
    f = m + (1.0 / lam) * np.log(np.sum(z))
    return -f  # we minimize -r_soft? careful: we want maximize r_soft.


def _neg_softmin(pts, lam):
    """Minimize this. = -r_soft, where r_soft = softmin(terms)."""
    pts = np.asarray(pts)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    z = np.exp(-lam * (terms - m))
    r_soft = m + (1.0 / lam) * np.log(np.sum(z))
    return -r_soft


def _neg_softmin_grad(pts, lam):
    """Gradient of -r_soft wrt pts. r_soft = softmin(terms).
    dr_soft/dx = sum_j w_j * d(terms_j)/dx, where w_j = exp(-lam t_j)/sum exp(-lam t_k).
    So d(-r_soft)/dx = - sum_j w_j * d(terms_j)/dx.
    """
    pts = np.asarray(pts)
    wall = 0.5 - np.max(np.abs(pts), axis=1)  # (N,)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0  # (N,N)
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    e = np.exp(-lam * (terms - m))
    w = e / np.sum(e)  # weights, len = N + N*(N-1)

    # d terms / d pts
    # wall part: wall_i = 0.5 - max(|x_i|,|y_i|)
    # d wall_i / d x_i = - sign(x_i) if |x_i|>|y_i| else 0 (and similar y)
    g_wall = np.zeros_like(pts)
    ax = np.abs(pts[:, 0]); ay = np.abs(pts[:, 1])
    mx = ax > ay
    g_wall[mx, 0] = -np.sign(pts[mx, 0])
    my = ay > ax
    g_wall[my, 1] = -np.sign(pts[my, 1])
    # where tie, split
    tie = (ax == ay) & (ax > 0)
    g_wall[tie, 0] = -0.5 * np.sign(pts[tie, 0])
    g_wall[tie, 1] = -0.5 * np.sign(pts[tie, 1])

    # pair part: pair_{i,j} = d_{ij}/2, d = |p_i - p_j|
    # d pair_{ij} / d p_i = (p_i - p_j)/(2 d_ij)
    # accumulate: for each i, sum over j != i w_{ij} * (p_i - p_j)/(2 d_ij)
    w_pair = w[N:].reshape(N, N)
    # set diagonal 0
    w_pair = w_pair * (1.0 - np.eye(N))
    # grad contribution to p_i: sum_j w_{ij} * (p_i - p_j)/(2 d_ij)
    # vectorized:
    diff = pts[:, None, :] - pts[None, :, :]  # (N,N,2)
    dmat = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    # factor = w_pair / (2 dmat)
    factor = w_pair / (2.0 * dmat)
    g_pair = (factor[..., None] * diff).sum(axis=1)  # (N,2)

    g_terms = g_wall + g_pair
    return -g_terms


def _hex_init():
    """Hexagonal (triangular) lattice init inside the square, trimmed to N."""
    # spacing chosen so that ~N circles fit with margin
    # triangular lattice: rows offset; vertical spacing = s*sqrt(3)/2
    # Try a range of spacings and pick the one that fits N points inside.
    best = None
    for s in np.linspace(0.04, 0.09, 200):
        vstep = s * np.sqrt(3) / 2.0
        # generate lattice
        pts = []
        x = -0.5 + s / 2.0
        row = 0
        y = -0.5 + vstep / 2.0
        while y <= 0.5:
            offset = (s / 2.0) if (row % 2) else 0.0
            xx = x + offset
            while xx <= 0.5:
                pts.append((xx, y))
                xx += s
            y += vstep
            row += 1
        pts = np.array(pts)
        if len(pts) >= N:
            # take a centered subset: sort by distance from origin
            r2 = (pts ** 2).sum(1)
            idx = np.argsort(r2)
            pts = pts[idx[:N]]
            # check validity
            if _score(pts) > 0:
                if best is None or _score(pts) > best[0]:
                    best = (_score(pts), pts.copy())
    if best is None:
        # fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best[1]


def _bounds():
    lo = np.full(N * 2, -0.5)
    hi = np.full(N * 2, 0.5)
    return (lo, hi)


def construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN

    pts = _hex_init().astype(np.float64)
    best = pts.copy()
    best_score = _score(best)

    lo, hi = _bounds()

    # Anneal softness: start coarse (small lam -> smooth) then refine (large lam -> sharp)
    lam_schedule = [40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0]

    cur = best.copy()
    for lam in lam_schedule:
        if time.time() > limit:
            break
        # run L-BFGS-B on the smooth surrogate
        def f(x):
            return _neg_softmin(x.reshape(N, 2), lam)

        def g(x):
            return _neg_softmin_grad(x.reshape(N, 2), lam).ravel()

        try:
            res = minimize(
                f, cur.ravel(), jac=g, method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 400, "maxfun": 2000, "ftol": 1e-10, "gtol": 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best = cand.copy()
                best_score = s
            cur = cand
        except Exception:
            pass

    # Final tight polish with very sharp lam
    if time.time() < limit:
        for lam in [4096.0, 8192.0, 16384.0]:
            if time.time() > limit:
                break
            def f(x):
                return _neg_softmin(x.reshape(N, 2), lam)

            def g(x):
                return _neg_softmin_grad(x.reshape(N, 2), lam).ravel()

            try:
                res = minimize(
                    f, best.ravel(), jac=g, method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"maxiter": 600, "maxfun": 3000, "ftol": 1e-12, "gtol": 1e-10},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best = cand.copy()
                    best_score = s
            except Exception:
                pass

    # Clamp to bounds for safety (numerical drift)
    best = np.clip(best, -0.5, 0.5)
    return best


# Self-test when run directly
if __name__ == "__main__":
    import time
    t = time.time()
    p = construct()
    print("elapsed", time.time() - t, "score", _score(p), "shape", p.shape)
