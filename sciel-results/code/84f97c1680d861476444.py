"""Physics-based relaxation for packing 50 equal circles in the unit disk."""
import numpy as np
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s wall-clock kill
_start = time.time()


def _remaining():
    return DEADLINE - (time.time() - _start)


def _pairwise_half_dists(p):
    # p: (N,2). Return (N, N) matrix of half-distances between centers.
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return d / 2.0


def _wall(p):
    return 1.0 - np.linalg.norm(p, axis=1)


def _score(p):
    # Hard min radius for a candidate.
    d = _pairwise_half_dists(p)
    np.fill_diagonal(d, np.inf)
    pair = d.min()
    w = _wall(p).min()
    return min(pair, w)


def _soft_min_surrogate(p, beta):
    # Return (value, grad) of a smooth surrogate that approximates -min(...).
    # We maximize r = min(pair, wall). Minimize f = -r.
    # soft-min via log-sum-exp: r_soft = -(1/beta) log sum exp(-beta * x_i)
    # where x_i are all candidate radii (pairwise + wall).
    d = _pairwise_half_dists(p)
    # exclude diagonal (infinite)
    mask = ~np.eye(N, dtype=bool)
    xd = d[mask]
    xw = _wall(p)
    x = np.concatenate([xd, xw])
    # numerical safety: clip very small negative values
    x = np.maximum(x, -10.0)
    # log-sum-exp stable
    m = np.min(x)
    s = np.sum(np.exp(-beta * (x - m)))
    r_soft = m - np.log(s) / beta
    # gradient of r_soft w.r.t. p
    # d r_soft / d x_i = exp(-beta*(x_i - m)) / s
    wts = np.exp(-beta * (x - m)) / s
    # gradient of -r_soft
    # d x_i / d p
    # For pairwise: x_ij = 0.5*||p_i - p_j||, d x_ij/d p_i = (p_i - p_j)/||p_i-p_j||
    # For wall: x_i = 1 - ||p_i||, d x_i/d p_i = -p_i/||p_i||
    grad = np.zeros_like(p)
    wd = wts[:N * (N - 1)]
    ww = wts[N * (N - 1):]
    # pairwise contribution
    # vectorize: for each pair (i,j), i<j
    # build index arrays
    ii, jj = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    diff = p[ii] - p[jj]
    dist = np.linalg.norm(diff, axis=1)
    dist = np.maximum(dist, 1e-12)
    unit = diff / dist[:, None]
    # x_ij = 0.5*dist, d x_ij/d p_i = 0.5 * unit, d x_ij/d p_j = -0.5 * unit
    # contribution to grad of -r_soft = -sum wts * d x/d p
    # we accumulate -wts * (d x/d p)
    contrib_i = -0.5 * wd[:, None] * unit
    contrib_j = -0.5 * wd[:, None] * unit
    np.add.at(grad, ii, contrib_i)
    np.add.at(grad, jj, contrib_j)
    # wall contribution
    rnorm = np.linalg.norm(p, axis=1)
    rnorm = np.maximum(rnorm, 1e-12)
    unit_w = p / rnorm[:, None]
    # x_w = 1 - rnorm, d x_w/d p = -unit_w
    # contribution to grad of -r_soft = -wts * (-unit_w) = +wts*unit_w... 
    # wait: we want grad of f = -r_soft. df/dp = -dr_soft/dp.
    # dr_soft/dp = sum wts_i * dx_i/dp. For wall: wts * (-unit_w).
    # So df/dp_wall = -wts * (-unit_w) = wts * unit_w.
    grad += ww[:, None] * unit_w
    return -r_soft, grad


def _lgbeta(p, beta, maxiter):
    from scipy.optimize import minimize
    res = minimize(_soft_min_surrogate, p, jac=True, method='L-BFGS-B',
                   args=(beta,),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                            'maxls': 50, 'maxfun': 2000})
    return res.x, res.fun


def _anneal(p0, betas, maxiters):
    p = p0
    for b, mi in zip(betas, maxiters):
        if _remaining() < 8.0:
            break
        p, _ = _lgbeta(p, b, mi)
    return p


def _hex_seed(spacing, jitter):
    rng = np.random.default_rng()
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing * 0.6) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) == 0:
        return None
    # take N closest to origin
    r = np.linalg.norm(pts, axis=1)
    order = np.argsort(r, kind='stable')
    pts = pts[order[:N]]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return pts


def construct():
    global _start
    _start = time.time()
    rng = np.random.default_rng(42)

    best_p = None
    best_r = -1.0

    # candidate starting spacings (denser = closer to optimal, but must fit N points)
    spacings = [0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
    # betas for annealing: low to high
    betas = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    maxiters = [40, 60, 80, 100, 120, 150]

    for sp in spacings:
        if _remaining() < 15.0:
            break
        for jitter in [0.0, 0.01, 0.02, 0.03]:
            if _remaining() < 15.0:
                break
            p0 = _hex_seed(sp, jitter)
            if p0 is None or len(p0) != N:
                continue
            # ensure valid start (inside, no overlap ideally)
            r0 = _score(p0)
            if r0 < 0.05:
                continue
            p = _anneal(p0, betas, maxiters)
            r = _score(p)
            # validity check
            if not np.all(np.isfinite(p)):
                continue
            if np.any(np.linalg.norm(p, axis=1) > 1.0 + 1e-9):
                continue
            if r > best_r:
                best_r = r
                best_p = p.copy()

    # Final high-precision polish on best result
    if best_p is not None and _remaining() > 10.0:
        for b, mi in [(1500.0, 200), (3000.0, 200), (6000.0, 200)]:
            if _remaining() < 8.0:
                break
            best_p, _ = _lgbeta(best_p, b, mi)
            r = _score(best_p)
            if not np.all(np.isfinite(best_p)) or np.any(np.linalg.norm(best_p, axis=1) > 1.0 + 1e-9):
                break

    if best_p is None:
        # fallback: simple hex
        p0 = _hex_seed(0.28, 0.0)
        return p0

    return best_p


if __name__ == '__main__':
    p = construct()
    print("radius:", _score(p))
    print("shape:", p.shape)
