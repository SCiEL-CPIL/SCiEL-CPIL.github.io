"""Packing 50 equal circles in unit square: epigraph formulation with smooth penalties."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
NP = len(IU[0])


def _score(pts):
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                      np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return min(float(wall.min()), float(dist[IU].min()) * 0.5)


def _softmin_all(x, beta):
    """Return (soft_min_value, weight_array) for wall and pair distances."""
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
    return val, sw


def _f_and_g(x, beta):
    """Objective: -soft_min. Gradient via chain rule."""
    val, w = _softmin_all(x, beta)
    p = x.reshape(N, 2)
    g = np.zeros_like(x)
    
    # Wall part: gradient of soft_min w.r.t. p
    wl = p[:, 0] + HALF
    wr = HALF - p[:, 0]
    wb = p[:, 1] + HALF
    wt = HALF - p[:, 1]
    wvals = np.stack([wl, wr, wb, wt], axis=1)
    mn = np.argmin(wvals, axis=1)
    for i in range(N):
        wi = w[i]
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
    
    # Pair part
    for k in range(NP):
        a, b = IU[0][k], IU[1][k]
        wk = w[N + k]
        if wk < 1e-14:
            continue
        diff = p[a] - p[b]
        dist = np.linalg.norm(diff)
        if dist < 1e-14:
            continue
        grad_unit = diff / (2.0 * dist)
        g[2*a:2*a+2] += wk * grad_unit
        g[2*b:2*b+2] -= wk * grad_unit
    
    return -val, -g


def _sa_push(pts0, target, tmax, rng, steps=2500):
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
    T0, T1 = 0.05, 5e-5
    cool = (T1 / T0) ** (1.0 / max(steps - 1, 1))
    
    for s in range(steps):
        if time.time() > tmax:
            break
        T = T0 * cool ** s
        nm = int(rng.integers(1, 4))
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


def _lbfgs_run(x0, beta, tmax, maxit=400):
    try:
        r = minimize(lambda x: _f_and_g(x, beta)[0], x0,
                     jac=lambda x: _f_and_g(x, beta)[1],
                     method='L-BFGS-B',
                     bounds=[(-HALF, HALF)] * (2 * N),
                     options={'maxiter': maxit, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40})
        if time.time() < tmax:
            return np.clip(r.x, -HALF, HALF)
    except Exception:
        pass
    return x0


def _init_hex(rng, jit=0.0):
    pts = []
    a = 0.115
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
    if jit > 0:
        pts += rng.normal(0, jit, pts.shape)
    return np.clip(pts, -HALF, HALF)


def _init_grid(rng, jit=0.0):
    m = 7
    tk = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in tk for x in tk])
    g = np.vstack([g, [[0.0, 0.0]]])[:N]
    if jit > 0:
        g += rng.normal(0, jit, g.shape)
    return np.clip(g, -HALF, HALF)


def _init_rand(rng):
    return rng.uniform(-HALF + 0.04, HALF - 0.04, (N, 2))


def _init_sym(rng):
    """5x10 grid with staggered columns."""
    pts = []
    for i in range(5):
        for j in range(10):
            x = (j + 0.5) / 10.0 - 0.5
            y = (i + 0.5) / 5.0 - 0.5
            if i % 2 == 1:
                x += 0.025
            pts.append([x, y])
    pts = np.array(pts[:N])
    pts += rng.normal(0, 0.015, pts.shape)
    return np.clip(pts, -HALF, HALF)


def construct():
    t0 = time.time()
    dl = t0 + 144.0
    rng = np.random.default_rng(2024)
    
    bp = None
    br = 0.0
    
    def improve(p0, rt, off=4.0):
        nonlocal bp, br
        if time.time() > dl - off:
            return
        sa_out, _ = _sa_push(p0, rt, dl - off, rng, steps=2000)
        pf = sa_out
        for b in [80.0, 250.0, 700.0]:
            if time.time() > dl - off:
                break
            pf = _lbfgs_run(pf, b, dl - off, maxit=300)
        r = _score(pf)
        if r > br:
            br = r
            bp = pf.copy()
    
    # Phase 1: diverse starts
    starts = [
        _init_hex(rng), _init_hex(rng, 0.01), _init_hex(rng, 0.025),
        _init_grid(rng), _init_grid(rng, 0.02), _init_grid(rng, 0.05),
        _init_rand(rng), _init_rand(rng),
        _init_sym(rng), _init_sym(rng),
    ]
    for s in starts:
        if time.time() > dl - 35.0:
            break
        improve(s, 0.067, off=5.0)
    
    if bp is None:
        m = 8
        tk = (np.arange(m) + 0.5) / m - 0.5
        return np.array([(x, y) for y in tk for x in tk])[:N]
    
    # Phase 2: binary search for radius
    if time.time() < dl - 25.0:
        lo, hi = br, 0.07138
        for _ in range(7):
            if time.time() > dl - 12.0:
                break
            mid = (lo + hi) * 0.5
            if time.time() > dl - 10.0:
                break
            sa_out, _ = _sa_push(bp, mid, dl - 8.0, rng, steps=1800)
            pf = sa_out
            for b in [150.0, 500.0]:
                if time.time() > dl - 6.0:
                    break
                pf = _lbfgs_run(pf, b, dl - 6.0, maxit=250)
            r = _score(pf)
            if r >= mid - 1e-9:
                lo, bp = mid, pf.copy()
                br = r
            else:
                hi = mid
    
    # Phase 3: high-precision soft-min polish
    if time.time() < dl - 6.0:
        for b in [40.0, 120.0, 350.0, 900.0]:
            if time.time() > dl - 3.0:
                break
            try:
                r = minimize(lambda x: _f_and_g(x, b)[0], bp.ravel(),
                             jac=lambda x: _f_and_g(x, b)[1],
                             method='L-BFGS-B',
                             bounds=[(-HALF, HALF)] * (2 * N),
                             options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13})
                ps = np.clip(r.x, -HALF, HALF)
                rs = _score(ps)
                if rs > br:
                    br, bp = rs, ps
            except Exception:
                pass
    
    # Phase 4: final nudge
    if time.time() < dl - 3.0:
        improve(bp, br + 5e-5, off=2.0)
    
    return bp
