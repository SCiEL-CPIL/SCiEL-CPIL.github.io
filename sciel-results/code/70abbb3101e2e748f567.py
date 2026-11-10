"""Force-directed repulsive optimization with smooth soft-max surrogate and local polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE = time.time() + 125.0  # safety margin before 150s kill


def _score(pts):
    """Exact radius for a set of centres (higher better)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # boundary margin
    m = HALF - np.abs(pts)
    bmin = m.min()
    if bmin <= 0:
        return -1.0
    # pairwise min distance
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pmin = d.min()
    return min(pmin / 2.0, bmin)


def _softmax_violation(pts, t, beta):
    """Soft-max of violations: pair: t - d_ij/2 ; boundary: t - (HALF-|c|).
    Returns (value, grad). Lower is better (<=0 means feasible at radius t)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    # pairwise
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    vp = t - d / 2.0  # violation per pair (i<j effectively, symmetric)
    # boundary
    mb = HALF - np.abs(pts)
    vb = t - mb  # shape (n,3)

    # combine all violations into one flat array for log-sum-exp
    # we'll do log-sum-exp over pairs (upper triangle) and boundary
    iu = np.triu_indices(n, k=1)
    vflat = np.concatenate([vp[iu], vb.ravel()])
    # log-sum-exp
    vmax = vflat.max()
    if not np.isfinite(vmax):
        return 0.0, np.zeros_like(pts)
    s = np.exp(vflat - vmax)
    lse = vmax + np.log(s.sum())
    w = s / s.sum()  # softmax weights

    # gradient wrt pts
    grad = np.zeros_like(pts)
    # pair part
    wpair = w[:len(iu)]
    # d(vp)/d(pts) for pair (i,j): vp = t - d/2, d(vp)/d c_i = -(c_i-c_j)/d, times 1/2 -> -(c_i-c_j)/(2d)
    # contribution to grad = wpair * d(vp)/d c
    for k in range(len(iu)):
        i, j = iu[0][k], iu[1][k]
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        g = -0.5 * diff / dist
        grad[i] += wpair[k] * g
        grad[j] -= wpair[k] * g
    # boundary part: vp_b = t - (HALF - |c_k|) = t - HALF + |c_k|
    # d/d c = sign(c_k)
    wbound = w[len(iu):].reshape(n, 3)
    grad += wbound * np.sign(pts)
    return lse, grad


def _optimize_from_start(start, t_init, beta, time_budget):
    """Gradient descent on soft-max violation with adaptive target radius."""
    pts = np.array(start, dtype=float)
    best = _score(pts)
    best_pts = pts.copy()
    t = t_init
    step = 0.02
    deadline = time.time() + time_budget
    n = len(pts)
    it = 0
    while time.time() < deadline:
        val, grad = _softmax_violation(pts, t, beta)
        # normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm
        # take a step
        new_pts = pts - step * grad
        # clamp inside cube with small margin
        new_pts = np.clip(new_pts, -HALF + 1e-6, HALF - 1e-6)
        # evaluate true score
        sc = _score(new_pts)
        if sc > best:
            best = sc
            best_pts = new_pts.copy()
            # raise target if feasible
            if val < 1e-3:
                t = min(t + 0.002, 0.12)
            step = min(step * 1.05, 0.05)
        else:
            # reduce step, maybe lower target
            step *= 0.7
            if val > 0.5:
                t = max(t - 0.002, 0.08)
        pts = new_pts
        it += 1
        # periodically try a local polish on true objective
        if it % 200 == 0 and time.time() < deadline - 5:
            polished = _polish(best_pts)
            ps = _score(polished)
            if ps > best:
                best = ps
                best_pts = polished
    return best, best_pts


def _polish(pts, max_iter=200):
    """Local refinement on the true nonsmooth objective via coordinate + small random kicks."""
    pts = np.array(pts, dtype=float)
    best = _score(pts)
    best_pts = pts.copy()
    for _ in range(max_iter):
        # random small perturbation of a few points
        idx = np.random.choice(N, size=min(5, N), replace=False)
        cand = best_pts.copy()
        cand[idx] += np.random.randn(len(idx), 3) * 0.002
        cand = np.clip(cand, -HALF + 1e-7, HALF - 1e-7)
        sc = _score(cand)
        if sc > best:
            best = sc
            best_pts = cand
    # final exact local solve with scipy on a smooth-ish proxy near optimum
    try:
        def neg_soft(x, t=best, beta=50.0):
            v, _ = _softmax_violation(x.reshape(N, 3), t, beta)
            return v
        res = minimize(neg_soft, best_pts.ravel(), method='BFGS',
                       options={'maxiter': 50, 'gtol': 1e-8})
        cand = np.clip(res.x.reshape(N, 3), -HALF + 1e-7, HALF - 1e-7)
        sc = _score(cand)
        if sc > best:
            best = sc
            best_pts = cand
    except Exception:
        pass
    return best_pts


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = 0.0
    best_pts = None

    # initial: simple grid as fallback
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid0 = grid[:N]
    best = _score(grid0)
    best_pts = grid0.copy()

    # several restarts from jittered grids
    n_restarts = 6
    per_budget = 15.0
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        # jittered grid start
        start = grid0 + rng.normal(0, 0.03, size=(N, 3))
        start = np.clip(start, -HALF + 1e-4, HALF - 1e-4)
        # ensure distinct
        sc0 = _score(start)
        if sc0 <= 0:
            start = grid0.copy()
        budget = min(per_budget, DEADLINE - time.time() - 8)
        if budget <= 2:
            break
        t_init = max(0.09, sc0 - 0.005)
        sc, pts = _optimize_from_start(start, t_init, beta=30.0, time_budget=budget)
        if sc > best:
            best = sc
            best_pts = pts
        # polish the best so far
        if time.time() < DEADLINE - 6:
            polished = _polish(best_pts)
            ps = _score(polished)
            if ps > best:
                best = ps
                best_pts = polished

    # final polish
    if time.time() < DEADLINE - 2:
        polished = _polish(best_pts)
        ps = _score(polished)
        if ps > best:
            best = ps
            best_pts = polished

    # final safety clamp
    best_pts = np.clip(best_pts, -HALF + 1e-7, HALF - 1e-7)
    return best_pts
