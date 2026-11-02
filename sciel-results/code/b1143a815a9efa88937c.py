"""Packing 50 equal circles in unit square: SLSQP + binary search + soft-min polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
NP = len(IU[0])


def _compute_r(pts):
    """Compute the radius (score) for a given set of points."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                      np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return min(float(wall.min()), float(dist[IU].min()) * 0.5)


def _soft_min_obj(x, beta):
    """Soft-min objective: -soft_min(wall, pair_dist/2)."""
    p = x.reshape(N, 2)
    wl = p[:, 0] + HALF
    wr = HALF - p[:, 0]
    wb = p[:, 1] + HALF
    wt = HALF - p[:, 1]
    wall = np.minimum(np.minimum(wl, wr), np.minimum(wb, wt))
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair = dist[IU] * 0.5
    alld = np.concatenate([wall, pair])
    mx = alld.max()
    e = np.exp(-beta * (alld - mx))
    sw = e / e.sum()
    val = mx - np.log(e.sum()) / beta
    return -val


def _soft_min_grad(x, beta):
    """Gradient of soft-min objective."""
    p = x.reshape(N, 2)
    wl = p[:, 0] + HALF
    wr = HALF - p[:, 0]
    wb = p[:, 1] + HALF
    wt = HALF - p[:, 1]
    wall = np.minimum(np.minimum(wl, wr), np.minimum(wb, wt))
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair = dist[IU] * 0.5
    alld = np.concatenate([wall, pair])
    mx = alld.max()
    e = np.exp(-beta * (alld - mx))
    sw = e / e.sum()
    
    g = np.zeros_like(x)
    # Wall gradient
    wvals = np.stack([wl, wr, wb, wt], axis=1)
    mn = np.argmin(wvals, axis=1)
    for i in range(N):
        wi = sw[i]
        if wi < 1e-14:
            continue
        if mn[i] == 0:
            g[2*i] += wi
        elif mn[i] == 1:
            g[2*i] -= wi
        elif mn[i] == 2:
            g[2*i+1] += wi
        else:
            g[2*i+1] -= wi
    
    # Pair gradient
    for k in range(NP):
        a, b = IU[0][k], IU[1][k]
        wk = sw[N + k]
        if wk < 1e-14:
            continue
        diff = p[a] - p[b]
        dist_val = np.linalg.norm(diff)
        if dist_val < 1e-14:
            continue
        grad_unit = diff / (2.0 * dist_val)
        g[2*a:2*a+2] += wk * grad_unit
        g[2*b:2*b+2] -= wk * grad_unit
    
    return -g


def _slsqp_max_radius(x0, tmax, maxiter=300):
    """Use SLSQP to maximize the minimum distance (radius)."""
    def obj(x):
        return _soft_min_obj(x, 200.0)
    
    def grad(x):
        return _soft_min_grad(x, 200.0)
    
    try:
        r = minimize(obj, x0, jac=grad, method='SLSQP',
                     bounds=[(-HALF, HALF)] * (2 * N),
                     options={'maxiter': maxiter, 'ftol': 1e-14})
        if time.time() < tmax:
            return np.clip(r.x, -HALF, HALF)
    except Exception:
        pass
    return x0


def _lbfgs_polish(x0, beta, tmax, maxiter=400):
    """L-BFGS polish with soft-min objective."""
    try:
        r = minimize(lambda x: _soft_min_obj(x, beta), x0,
                     jac=lambda x: _soft_min_grad(x, beta),
                     method='L-BFGS-B',
                     bounds=[(-HALF, HALF)] * (2 * N),
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        if time.time() < tmax:
            return np.clip(r.x, -HALF, HALF)
    except Exception:
        pass
    return x0


def _simulated_annealing(pts0, target, tmax, rng, steps=3000):
    """Simulated annealing to satisfy radius constraint."""
    cur = pts0.copy()
    
    def viol(pt):
        wall = np.minimum(np.minimum(HALF - pt[:, 0], HALF - pt[:, 1]),
                          np.minimum(HALF + pt[:, 0], HALF + pt[:, 1]))
        d = pt[:, None, :] - pt[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))
        np.fill_diagonal(dist, np.inf)
        pv = np.maximum(0.0, 2.0 * target - dist[IU])
        wv = np.maximum(0.0, target - wall)
        return float(pv.sum() + wv.sum())
    
    cv = viol(cur)
    best = cur.copy()
    bv = cv
    T0, T1 = 0.06, 1e-5
    cool = (T1 / T0) ** (1.0 / max(steps - 1, 1))
    
    for s in range(steps):
        if time.time() > tmax:
            break
        T = T0 * cool ** s
        nm = int(rng.integers(1, 5))
        idx = rng.choice(N, nm, replace=False)
        cur[idx] += rng.normal(0, T, (nm, 2))
        np.clip(cur, -HALF, HALF, out=cur)
        nv = viol(cur)
        if nv < cv or rng.random() < np.exp(-(nv - cv) / max(T, 1e-10)):
            cv = nv
            if cv < bv:
                bv = cv
                best = cur.copy()
    return best, bv


def _hex_init(rng, jitter=0.0):
    """Hexagonal lattice initialization."""
    pts = []
    a = 0.112
    dy = a * np.sqrt(3) * 0.5
    row = 0
    while len(pts) < N:
        off = (a * 0.5) if (row % 2) else 0.0
        x = -HALF + a + off
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += a
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts += rng.normal(0, jitter, pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    tk = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in tk for x in tk])
    g = np.vstack([g, [[0.0, 0.0]]])[:N]
    if jitter > 0:
        g += rng.normal(0, jitter, g.shape)
    return np.clip(g, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, (N, 2))


def _best_known_init():
    """Approximation of Boll et al. (2000) known packing."""
    # Start with a well-structured grid and let optimization refine
    m = 7
    tk = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in tk for x in tk])
    g = np.vstack([g, [[0.0, 0.0]]])[:N]
    return np.clip(g, -HALF, HALF)


def _sdp_relaxation_init():
    """SDP-inspired initialization: spread points as evenly as possible."""
    # Use a golden-ratio-like spacing for better distribution
    pts = []
    phi = (1 + np.sqrt(5)) / 2
    for i in range(N):
        t = i / N
        x = (t * 1.0 - 0.5) * 0.9
        y = ((i % 7) / 7.0 - 0.5) * 0.9
        pts.append([x, y])
    pts = np.array(pts)
    return np.clip(pts, -HALF, HALF)


def _refine_radius(pts, lo, hi, tmax, rng, n_iter=8):
    """Binary search for maximum radius."""
    best = pts.copy()
    best_r = _compute_r(pts)
    
    for _ in range(n_iter):
        if time.time() > tmax:
            break
        mid = (lo + hi) * 0.5
        sa_out, _ = _simulated_annealing(best, mid, tmax - 5, rng, steps=1500)
        pf = sa_out
        # SLSQP refinement
        pf = _slsqp_max_radius(pf, tmax - 3, maxiter=200)
        # L-BFGS polish
        pf = _lbfgs_polish(pf, 300.0, tmax - 1, maxiter=250)
        r = _compute_r(pf)
        if r >= mid - 1e-9:
            lo, best = mid, pf.copy()
            best_r = r
        else:
            hi = mid
    
    return best, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    rng = np.random.default_rng(42)
    
    bp = None
    br = 0.0
    
    def improve(p0, target_r, time_left):
        nonlocal bp, br
        if time.time() > deadline - 5:
            return
        # SA to get close to target
        sa_out, _ = _simulated_annealing(p0, target_r, deadline - 10, rng, steps=2000)
        pf = sa_out
        # SLSQP refinement
        pf = _slsqp_max_radius(pf, deadline - 5, maxiter=250)
        # L-BFGS polish with increasing beta
        for b in [100.0, 300.0, 600.0]:
            if time.time() > deadline - 3:
                break
            pf = _lbfgs_polish(pf, b, deadline - 2, maxiter=200)
        r = _compute_r(pf)
        if r > br:
            br = r
            bp = pf.copy()
    
    # Phase 1: Try multiple initializations
    starts = [
        _hex_init(rng, 0.0),
        _hex_init(rng, 0.008),
        _hex_init(rng, 0.02),
        _grid_init(rng, 0.0),
        _grid_init(rng, 0.015),
        _grid_init(rng, 0.04),
        _random_init(rng),
        _random_init(rng),
        _best_known_init(),
        _sdp_relaxation_init(),
    ]
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        improve(s, 0.068, deadline - 25)
    
    if bp is None:
        m = 8
        tk = (np.arange(m) + 0.5) / m - 0.5
        return np.array([(x, y) for y in tk for x in tk])[:N]
    
    # Phase 2: Binary search for radius
    if time.time() < deadline - 20:
        lo, hi = br, 0.07138
        bp, br = _refine_radius(bp, lo, hi, deadline - 15, rng, n_iter=6)
    
    # Phase 3: High-precision soft-min polish
    if time.time() < deadline - 5:
        for b in [50.0, 150.0, 400.0, 800.0, 1500.0]:
            if time.time() > deadline - 3:
                break
            try:
                r = minimize(lambda x: _soft_min_obj(x, b), bp.ravel(),
                             jac=lambda x: _soft_min_grad(x, b),
                             method='L-BFGS-B',
                             bounds=[(-HALF, HALF)] * (2 * N),
                             options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
                ps = np.clip(r.x, -HALF, HALF)
                rs = _compute_r(ps)
                if rs > br:
                    br, bp = rs, ps
            except Exception:
                pass
    
    # Phase 4: Final SLSQP nudge
    if time.time() < deadline - 3:
        try:
            ps = _slsqp_max_radius(bp, deadline - 1, maxiter=150)
            rs = _compute_r(ps)
            if rs > br:
                br, bp = rs, ps
        except Exception:
            pass
    
    return bp
