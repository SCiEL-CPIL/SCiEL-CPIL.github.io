"""Improved program: smooth log-sum-exp surrogate + L-BFGS-B for 50-sphere packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_T0 = time.time()
_DEADLINE = 140.0  # safety margin below 150 s wall-clock limit


def _score(pts):
    """Exact radius: min(min_pair_dist/2, min_wall_dist)."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


def _logsumexp(x, beta):
    """Numerically stable soft-min: -(1/beta) log sum exp(-beta * x)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.min()
    return -m - np.log(np.sum(np.exp(-beta * (x - m)))) / beta


def _surrogate_loss(pts, beta):
    """
    Smooth surrogate to minimize:
      loss = softmin(pair_dists) + softmin(wall_dists)
    Minimizing this pushes all distances up (maximizes the bottleneck).
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Pairwise distances (upper triangle only)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu]
    # Wall distances
    wall = 0.5 - np.abs(pts)
    wall_d = wall.ravel()
    # Soft-min of both sets
    s1 = _logsumexp(pair_d, beta)
    s2 = _logsumexp(wall_d, beta)
    return s1 + s2


def _surrogate_grad(pts, beta):
    """Gradient of the surrogate loss w.r.t. pts."""
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)

    # --- pair soft-min gradient ---
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu]
    # weights w_i = exp(-beta * (x_i - m)) / sum(...)
    m_p = pair_d.min()
    e = np.exp(-beta * (pair_d - m_p))
    w_p = e / e.sum()  # shape (M,)
    # d/dx_i of softmin = -sum_j w_j * d x_j / d x_i
    # For pair (a,b): d pair_d / d pts[a] = (pts[a]-pts[b])/pair_d
    #                  d pair_d / d pts[b] = (pts[b]-pts[a])/pair_d
    g = np.zeros_like(pts)
    # accumulate over pairs
    a_idx = iu[0]
    b_idx = iu[1]
    # unit vectors
    da = pts[a_idx] - pts[b_idx]
    db = pts[b_idx] - pts[a_idx]
    inv_d = np.where(pair_d > 1e-15, 1.0 / pair_d, 0.0)
    w = w_p * inv_d
    # d softmin / d pts[a] = -sum w_j * da_j  (da_j is (pts[a]-pts[b])/pair_d)
    ga = np.zeros_like(pts)
    gb = np.zeros_like(pts)
    # use np.add.at for repeated indices
    np.add.at(ga, a_idx, w[:, None] * da)
    np.add.at(gb, b_idx, w[:, None] * db)
    g -= (ga + gb)

    # --- wall soft-min gradient ---
    wall = 0.5 - np.abs(pts)
    wall_d = wall.ravel()
    m_w = wall_d.min()
    ew = np.exp(-beta * (wall_d - m_w))
    ww = ew / ew.sum()  # shape (3N,)
    # d wall_d / d pts: for element (j,k), wall = 0.5 - |pts[j,k]|
    # d/dpts[j,k] = -sign(pts[j,k])
    signs = -np.sign(pts)  # shape (N,3)
    g_w = (ww.reshape(N, 3)) * signs
    g -= g_w

    return g


def _lbgfsb_polish(pts, beta, iters_max, time_budget):
    """Run L-BFGS-B on the smooth surrogate, respecting time budget."""
    pts = np.asarray(pts, dtype=np.float64).copy()
    x0 = pts.ravel()

    def fun(x):
        return _surrogate_loss(x.reshape(N, 3), beta)

    def jac(x):
        return _surrogate_grad(x.reshape(N, 3), beta).ravel()

    best_x = x0.copy()
    best_f = fun(x0)

    def cb(xk):
        nonlocal best_x, best_f
        f = fun(xk)
        if f < best_f:
            best_f = f
            best_x = xk.copy()
        if time.time() > time_budget:
            raise RuntimeError("timeout")

    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (N * 3),
                       options={'maxiter': iters_max, 'maxfun': iters_max * 4,
                                'ftol': 1e-12, 'gtol': 1e-10, 'callback': cb})
        if res.x is not None:
            f = fun(res.x)
            if f < best_f:
                best_f = f
                best_x = res.x.copy()
    except RuntimeError:
        pass
    except Exception:
        pass
    return best_x.reshape(N, 3)


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with adaptive kicks."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.015 * (T / T0 + 0.05)
        n_move = max(1, N // 4)
        idx = rng.choice(N, size=n_move, replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand = np.clip(cand, -0.5 + 1e-12, 0.5 - 1e-12)
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 400 == 0 and time.time() > _DEADLINE - 5:
            break
    return best, best_r


def _make_starts(rng):
    starts = []
    # 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    starts.append(grid[:N].copy())

    # 5x5x2 slab
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    if len(slab) >= N:
        starts.append(slab[:N].copy())

    # Random jittered grids
    for _ in range(4):
        s = starts[0] + rng.normal(0.0, 0.03, size=starts[0].shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)
    return starts


def construct():
    rng = np.random.default_rng(777)
    starts = _make_starts(rng)

    best = starts[0].copy()
    best_r = _score(best)

    # Phase 1: anneal each start, then polish with L-BFGS-B
    for s in starts:
        if time.time() > _DEADLINE - 30:
            break
        a, ar = _anneal(s, T0=0.04, T1=0.0005, steps=2500, rng=rng)
        if ar > best_r:
            best_r, best = ar, a

        # Polish with L-BFGS-B using a moderate beta
        if time.time() > _DEADLINE - 25:
            break
        p = _lbgfsb_polish(a, beta=200.0, iters_max=200, time_budget=_DEADLINE - 20)
        p = np.clip(p, -0.5 + 1e-12, 0.5 - 1e-12)
        pr = _score(p)
        if pr > best_r:
            best_r, best = pr, p

    # Phase 2: high-beta polish on the current best
    if time.time() < _DEADLINE - 10:
        for beta in [400.0, 800.0]:
            if time.time() > _DEADLINE - 8:
                break
            p = _lbgfsb_polish(best, beta=beta, iters_max=300, time_budget=_DEADLINE - 5)
            p = np.clip(p, -0.5 + 1e-12, 0.5 - 1e-12)
            pr = _score(p)
            if pr > best_r:
                best_r, best = pr, p

    # Final safety: ensure strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)

    return best
