"""Fresh design: smooth soft-min radius optimization + SLSQP polish for 50-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0          # hard wall-clock limit
SAFETY_MARGIN = 15.0        # return best before this margin
START_TIME = time.time()

def _remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _score(c):
    """True objective: largest common radius (disjoint + inside)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(c)):
        return -np.inf
    # wall margins: distance from each centre to the nearest wall
    walls = np.minimum(0.5 - np.abs(c), axis=1)
    r_wall = np.min(walls)
    # pairwise half-distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    r = min(r_wall, r_pair)
    if r <= 0:
        return -np.inf
    return float(r)

def _soft_radius(c, alpha):
    """Smooth surrogate of the true radius using log-sum-exp.
    Returns (value, grad). value approximates the min radius (negative log-sum-exp)."""
    c = np.asarray(c, dtype=float)
    # wall terms
    # For each centre, wall margin to each of 4 walls
    x = c[:, 0]; y = c[:, 1]
    # distance to left, right, bottom, top walls (as margins)
    wall_margins = np.stack([0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y], axis=1)  # (N,4)
    # pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    # half-distance terms (each pair once)
    iu = np.triu_indices(N, k=1)
    half_d = d[iu]  # (M,)
    # log-sum-exp over negative values: -logsumexp(-a_i) approximates min a_i
    # combine all "radius-like" quantities: half_d (pair) and wall_margins (wall)
    a = np.concatenate([half_d, wall_margins.ravel()])
    # soft-min via logsumexp: f = -logsumexp(-alpha*a)/alpha
    neg = -alpha * a
    maxneg = np.max(neg)
    lse = maxneg + np.log(np.sum(np.exp(neg - maxneg)))
    val = -lse / alpha
    # gradient wrt c
    # d f / d a_i = (1/alpha) * (exp(neg_i - lse))  ... but lse includes maxneg; recompute
    exps = np.exp(neg - lse)  # weights sum to 1
    # df/da = -exps/alpha  (since f = -lse/alpha, d lse/da_i = alpha*exp(...), so df/da_i = -exp(...))
    # Actually: f = -lse/alpha. d lse/d a_i = -alpha*exp(neg_i - lse). So d f/d a_i = -(-alpha*exp)/alpha = exp.
    # So df/da_i = exps_i.
    grad_a = exps  # shape (len(a),)
    # accumulate gradient wrt c
    grad = np.zeros_like(c)
    # pair part
    gpair = grad_a[:len(half_d)]
    # half_d = d_ij/2, d = |c_i-c_j|, d(d)/dc_i = (c_i-c_j)/d, d(half_d)/dc_i = (c_i-c_j)/(2d)
    for k in range(len(iu[0])):
        i = iu[0][k]; j = iu[1][k]
        dij = d[i, j]
        if dij < 1e-12:
            continue
        w = gpair[k] * 0.5 / dij
        grad[i] += w * (c[i] - c[j])
        grad[j] += w * (c[j] - c[i])
    # wall part
    gwall = grad_a[len(half_d):].reshape(N, 4)
    # wall_margins: col0=0.5-x, col1=0.5+x, col2=0.5-y, col3=0.5+y
    # d(0.5-x)/dx = -1 ; d(0.5+x)/dx=+1 ; d(0.5-y)/dy=-1 ; d(0.5+y)/dy=+1
    grad[:, 0] += -gwall[:, 0] + gwall[:, 1]
    grad[:, 1] += -gwall[:, 2] + gwall[:, 3]
    return val, grad

def _adam_smooth(c0, alpha, lr, iters, deadline_fn):
    """Adam ascent on the soft radius."""
    c = np.array(c0, dtype=float)
    m = np.zeros_like(c); v = np.zeros_like(c)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _score(c)
    for t in range(1, iters + 1):
        if deadline_fn() < 5.0:
            break
        val, g = _soft_radius(c, alpha)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        c = c + lr * mh / (np.sqrt(vh) + eps)
        # project into square with tiny margin to keep finite
        c = np.clip(c, -0.4999, 0.4999)
        s = _score(c)
        if s > best_val:
            best_val = s
    return c, best_val

def _slsqp_polish(c0):
    """SLSQP polish maximizing true radius with contact/wall constraints."""
    c = np.array(c0, dtype=float).reshape(-1)
    n = c.size
    def neg_radius(v):
        cc = v.reshape(N, 2)
        # wall
        walls = np.minimum(0.5 - np.abs(cc), axis=1)
        rw = np.min(walls)
        diff = cc[:, None, :] - cc[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        rp = np.min(d) / 2.0
        return -min(rw, rp)
    # constraints: keep all centres inside (hard) and pairwise distance >= 2*target (soft via inequality on min)
    # We'll use inequality constraints that push min radius up: r_pair - r_target >= 0 etc.
    # Simpler: maximize -min(...) using SLSQP with bounds only; the objective is nonsmooth but SLSQP can handle.
    bounds = [(-0.5, 0.5)] * n
    try:
        res = minimize(neg_radius, c, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success or res.fun < neg_radius(c):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return c0

def _hex_init():
    """Hexagonal packing seed."""
    pts = []
    # hex lattice: spacing s, rows offset
    s = 0.2
    y = -0.45
    row = 0
    while y <= 0.45 and len(pts) < N:
        xoff = (row % 2) * s * 0.5
        x = -0.45 + xoff
        while x <= 0.45 and len(pts) < N:
            pts.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    arr = np.array(pts[:N], dtype=float)
    # center it
    arr -= arr.mean(axis=0)
    arr *= 0.9  # shrink to be safely inside
    return arr

def _grid_init(m):
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _jittered(base, rng, amp):
    return base + rng.normal(0, amp, size=base.shape)

def construct():
    global START_TIME
    START_TIME = time.time()
    rng = np.random.default_rng(12345)
    best_c = None
    best_s = -np.inf

    def consider(c):
        nonlocal best_c, best_s
        s = _score(c)
        if s > best_s:
            best_s = s
            best_c = np.array(c, dtype=float)

    # seed pool
    seeds = []
    m = int(np.ceil(np.sqrt(N)))
    seeds.append(_grid_init(m))
    seeds.append(_hex_init())
    seeds.append(_jittered(_grid_init(m), rng, 0.02))
    seeds.append(_jittered(_hex_init(), rng, 0.01))
    # rotated grid
    g = _grid_init(m)
    ang = 0.3
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    seeds.append(g @ R.T * 0.85)

    # Phase 1: smooth soft-min ascent from each seed (short), keep best
    for seed in seeds:
        if _remaining() < 10.0:
            break
        for alpha in [20.0, 40.0]:
            if _remaining() < 8.0:
                break
            c, s = _adam_smooth(seed, alpha, lr=0.02, iters=400, deadline_fn=_remaining)
            consider(c)

    # Phase 2: SLSQP polish from top candidates
    if _remaining() > 10.0 and best_c is not None:
        polished = _slsqp_polish(best_c)
        consider(polished)

    # Phase 3: a few more SLSQP polishes from jittered best
    if _remaining() > 15.0 and best_c is not None:
        for _ in range(3):
            if _remaining() < 10.0:
                break
            pert = _jittered(best_c, rng, 0.005)
            polished = _slsqp_polish(pert)
            consider(polished)

    if best_c is None:
        # fallback to plain grid
        return _grid_init(m)

    # final safety: ensure validity
    final = np.clip(best_c, -0.5, 0.5)
    return final
