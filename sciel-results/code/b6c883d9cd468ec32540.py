"""
Packing of 50 equal circles in a unit square.
Uses SA for global search, SLSQP for constraint enforcement, and L-BFGS with
soft-min/soft-violation surrogates for high-precision polishing.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 144.0

_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    if min_wall < -1e-12:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    c = x.reshape(-1, 2)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    wv = r_target - walls
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pv = 2.0 * r_target - dists
    zw = np.clip(beta * wv, -50, 50)
    zp = np.clip(beta * pv, -50, 50)
    return float(np.sum(np.log1p(np.exp(zw))) / beta + np.sum(np.log1p(np.exp(zp))) / beta)

def _soft_violation_grad(x, r_target, beta):
    c = x.reshape(-1, 2)
    g = np.zeros_like(x)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    wv = r_target - walls
    zw = np.clip(beta * wv, -50, 50)
    sw = 1.0 / (1.0 + np.exp(-zw))
    wx_lim = wx < wy; wy_lim = wy < wx; tie = ~(wx_lim | wy_lim)
    if np.any(wx_lim):
        g[2*np.nonzero(wx_lim)[0]] += sw[wx_lim] * np.sign(c[wx_lim, 0])
    if np.any(wy_lim):
        g[2*np.nonzero(wy_lim)[0]+1] += sw[wy_lim] * np.sign(c[wy_lim, 1])
    if np.any(tie):
        idx = np.nonzero(tie)[0]
        g[2*idx] += 0.5 * sw[tie] * np.sign(c[idx, 0])
        g[2*idx+1] += 0.5 * sw[tie] * np.sign(c[idx, 1])
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    ds = np.maximum(dists, 1e-12)
    pv = 2.0 * r_target - dists
    zp = np.clip(beta * pv, -50, 50)
    sp = 1.0 / (1.0 + np.exp(-zp))
    f = (sp / ds)[:, None] * diff
    np.add.at(g, 2*_IU[0], -f[:, 0])
    np.add.at(g, 2*_IU[0]+1, -f[:, 1])
    np.add.at(g, 2*_IU[1], f[:, 0])
    np.add.at(g, 2*_IU[1]+1, f[:, 1])
    return g

def _soft_min_obj(x, beta):
    c = x.reshape(-1, 2)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_d = dists / 2.0
    vals = np.concatenate([walls, half_d])
    z = np.clip(-beta * vals, -50, 50)
    z_max = np.max(z)
    lse = z_max + np.log(np.sum(np.exp(z - z_max)))
    return float(lse / beta)

def _soft_min_grad(x, beta):
    c = x.reshape(-1, 2)
    g = np.zeros_like(x)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_d = dists / 2.0
    vals = np.concatenate([walls, half_d])
    z = np.clip(-beta * vals, -50, 50)
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_e = np.sum(exp_z)
    w = exp_z / sum_e
    n_w = N
    ww = w[:n_w]
    wx_lim = wx < wy; wy_lim = wy < wx; tie = ~(wx_lim | wy_lim)
    if np.any(wx_lim):
        idx = np.nonzero(wx_lim)[0]
        g[2*idx] -= ww[wx_lim] * np.sign(c[idx, 0])
    if np.any(wy_lim):
        idx = np.nonzero(wy_lim)[0]
        g[2*idx+1] -= ww[wy_lim] * np.sign(c[idx, 1])
    if np.any(tie):
        idx = np.nonzero(tie)[0]
        g[2*idx] -= 0.5 * ww[tie] * np.sign(c[idx, 0])
        g[2*idx+1] -= 0.5 * ww[tie] * np.sign(c[idx, 1])
    pw = w[n_w:]
    ds = np.maximum(dists, 1e-12)
    f = (pw / ds)[:, None] * diff
    np.add.at(g, 2*_IU[0], f[:, 0])
    np.add.at(g, 2*_IU[0]+1, f[:, 1])
    np.add.at(g, 2*_IU[1], -f[:, 0])
    np.add.at(g, 2*_IU[1]+1, -f[:, 1])
    return g

def _initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
    rng = np.random.default_rng(12345)
    grid += rng.normal(0, 0.005, grid.shape)
    return np.clip(grid, -0.49, 0.49)

def _hex_init():
    best_grid = None; best_r = 0
    rng = np.random.default_rng(42)
    for attempt in range(10):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        sw = 1.0 / (cols + 0.5)
        sh = 1.0 / (rows * 0.866 + 0.5)
        s = min(sw, sh)
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols-1)/2) * s
                y = (j - (rows-1)/2) * s * 0.866
                if j % 2 == 1: x += s/2
                centers.append([x, y])
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r; best_grid = centers.copy()
    return best_grid if best_grid is not None else _initial_grid()

def _deformation_init():
    m = int(np.ceil(np.sqrt(N)))
    base = []
    s = 1.0 / (m + 0.3)
    for j in range(m):
        for i in range(m):
            x = (i - (m-1)/2) * s
            y = (j - (m-1)/2) * s * 0.866
            if j % 2 == 1: x += s/2
            base.append([x, y])
    base = np.array(base[:N])
    rng = np.random.default_rng(42)
    nm = 3
    fx = rng.uniform(1, 4, nm); fy = rng.uniform(1, 4, nm)
    ax = rng.uniform(0.01, 0.04, nm); ay = rng.uniform(0.01, 0.04, nm)
    px = rng.uniform(0, 2*np.pi, nm); py = rng.uniform(0, 2*np.pi, nm)
    d = base.copy()
    for k in range(nm):
        d[:, 0] += ax[k]*np.sin(fx[k]*base[:, 0]+px[k])*np.cos(fy[k]*base[:, 1]+py[k])
        d[:, 1] += ay[k]*np.sin(fy[k]*base[:, 1]+py[k])*np.cos(fx[k]*base[:, 0]+px[k])
    d += rng.normal(0, 0.003, d.shape)
    return np.clip(d, -0.49, 0.49)

def _simulated_annealing(centers, rng, time_budget):
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_r = _compute_radius(cur)
    T0 = 0.012; T1 = 0.00003
    n_steps = max(20000, min(80000, int(time_budget / 0.00035)))
    cool = (T1/T0)**(1.0/max(n_steps,1))
    T = T0
    iu0, iu1 = _IU[0], _IU[1]
    pairs_of = [[] for _ in range(N)]
    for p in range(_NPAIRS):
        pairs_of[iu0[p]].append(p); pairs_of[iu1[p]].append(p)
    ci = cur[iu0]; cj = cur[iu1]
    diff = ci - cj
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    wx = 0.5 - np.abs(cur[:, 0]); wy = 0.5 - np.abs(cur[:, 1])
    walls = np.minimum(wx, wy)
    for step in range(n_steps):
        if time.time() - start > time_budget: break
        mw = np.min(walls); mp = np.sqrt(np.min(dist_sq))/2.0
        cr = min(mp, mw)
        i = rng.integers(0, N)
        frac = T/T0
        ms = 0.003 + 0.007*frac
        if rng.random() < 0.08: ms *= 2.5
        d = rng.normal(size=2); n = np.linalg.norm(d)
        if n > 1e-12: d /= n
        else: d = np.array([1.0, 0.0])
        old = cur[i].copy()
        new = old + d*ms
        new = np.clip(new, -0.5, 0.5)
        owx = 0.5-abs(old[0]); owy = 0.5-abs(old[1]); ow = min(owx, owy)
        nwx = 0.5-abs(new[0]); nwy = 0.5-abs(new[1]); nw = min(nwx, nwy)
        walls[i] = nw
        for p in pairs_of[i]:
            if iu0[p] == i:
                j = iu1[p]; dd = new - cur[j]
            else:
                j = iu0[p]; dd = cur[j] - new
            dist_sq[p] = np.dot(dd, dd)
        tw = np.min(walls); tp = np.sqrt(np.min(dist_sq))/2.0
        tr = min(tp, tw)
        delta = tr - cr
        if delta >= 0:
            cur[i] = new
            if tr > best_r:
                best_r = tr; best = cur.copy()
        else:
            if rng.random() < np.exp(delta/max(T,1e-10)):
                cur[i] = new
            else:
                walls[i] = ow
                for p in pairs_of[i]:
                    if iu0[p] == i:
                        j = iu1[p]; dd = old - cur[j]
                    else:
                        j = iu0[p]; dd = cur[j] - old
                    dist_sq[p] = np.dot(dd, dd)
        T *= cool
    return best, best_r

def _slsqp_polish(centers, time_budget):
    """SLSQP with explicit constraints to enforce non-overlap and wall bounds."""
    start = time.time()
    x0 = centers.flatten()
    r0 = _compute_radius(centers)
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Maximize soft-min radius
        return -_soft_min_obj(x, 500.0)
    
    def objective_grad(x):
        return -_soft_min_grad(x, 500.0)
    
    # Constraints: g(x) >= 0
    # Wall constraints: 0.5 - |x_i| >= 0 and 0.5 - |y_i| >= 0
    # Pair constraints: dist_ij - 2*r0 >= 0
    
    def make_constraints(r_target):
        cons = []
        # Wall x
        for i in range(N):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 0.5 - abs(x[2*i]),
                         'jac': lambda x, i=i: np.array([0]*2*i + [-np.sign(x[2*i])] + [0]*(2*N-2*i-1))})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 0.5 - abs(x[2*i+1]),
                         'jac': lambda x, i=i: np.array([0]*(2*i) + [0] + [-np.sign(x[2*i+1])] + [0]*(2*N-2*i-2))})
        # Pair constraints
        iu0, iu1 = _IU[0], _IU[1]
        for p in range(_NPAIRS):
            i, j = iu0[p], iu1[p]
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - 2*r_target,
                         'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)})
        return cons
    
    def _pair_jac(x, i, j):
        g = np.zeros(2*N)
        dx = x[2*i] - x[2*j]
        dy = x[2*i+1] - x[2*j+1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < 1e-12: d = 1e-12
        g[2*i] = dx/d; g[2*i+1] = dy/d
        g[2*j] = -dx/d; g[2*j+1] = -dy/d
        return g
    
    try:
        result = opt.minimize(
            objective, x0, jac=objective_grad, method='SLSQP',
            bounds=[(-0.5, 0.5)]*(2*N),
            constraints=make_constraints(r0),
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        return polished
    except Exception:
        return centers

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    start = time.time()
    def og(x):
        if time.time() - start > time_budget: return 0.0, np.zeros_like(x)
        return _soft_violation_obj(x, r_target, beta), _soft_violation_grad(x, r_target, beta)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)]*(2*N),
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14,
                                    'maxfun': 25000, 'disp': False, 'maxls': 50})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_soft_min_polish(x0, beta, time_budget):
    start = time.time()
    def og(x):
        if time.time() - start > time_budget: return 0.0, np.zeros_like(x)
        return _soft_min_obj(x, beta), _soft_min_grad(x, beta)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)]*(2*N),
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13,
                                    'maxfun': 15000, 'disp': False})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    for i in range(22):
        if time.time() - start > time_budget: break
        mid = (lo + hi) / 2
        progress = i / 21
        beta = 200.0 + 800.0 * progress
        remaining = time_budget - (time.time() - start)
        if remaining < 5: break
        lb = min(18.0, remaining * 0.85)
        trial = _lbfgs_feasibility_polish(best.flatten(), mid, beta, lb)
        trial_r = _compute_radius(trial)
        if trial_r >= mid - 1e-8:
            lo = mid; best = trial; best_r = trial_r
        else:
            hi = mid
    return best, best_r

def _coordinate_descent_polish(centers, time_budget):
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_r = _compute_radius(cur)
    iu0, iu1 = _IU[0], _IU[1]
    pairs_of = [[] for _ in range(N)]
    for p in range(_NPAIRS):
        pairs_of[iu0[p]].append(p); pairs_of[iu1[p]].append(p)
    ci = cur[iu0]; cj = cur[iu1]
    diff = ci - cj
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    wx = 0.5 - np.abs(cur[:, 0]); wy = 0.5 - np.abs(cur[:, 1])
    walls = np.minimum(wx, wy)
    for it in range(100):
        if time.time() - start > time_budget: break
        improved = False
        for i in range(N):
            for coord in range(2):
                ov = cur[i, coord]
                for delta in [0.0005, 0.001, 0.002, 0.005, -0.0005, -0.001, -0.002, -0.005]:
                    nv = ov + delta
                    if nv < -0.5 or nv > 0.5: continue
                    if coord == 0:
                        nw = min(0.5 - abs(nv), wy[i])
                    else:
                        nw = min(wx[i], 0.5 - abs(nv))
                    owi = walls[i]; walls[i] = nw
                    for p in pairs_of[i]:
                        if iu0[p] == i:
                            j = iu1[p]
                            d = cur[i].copy(); d[coord] = nv; d = d - cur[j]
                        else:
                            j = iu0[p]
                            d = cur[j].copy(); d[coord] = nv; d = cur[i] - d
                        dist_sq[p] = np.dot(d, d)
                    tmw = np.min(walls); tmp = np.sqrt(np.min(dist_sq))/2.0
                    tr = min(tmp, tmw)
                    if tr > best_r + 1e-12:
                        cur[i, coord] = nv; best_r = tr; best = cur.copy()
                        improved = True; break
                    else:
                        walls[i] = owi
                        for p in pairs_of[i]:
                            if iu0[p] == i:
                                j = iu1[p]; d = cur[i] - cur[j]
                            else:
                                j = iu0[p]; d = cur[j] - cur[i]
                            dist_sq[p] = np.dot(d, d)
                if improved: break
            if improved: break
        if not improved: break
    return best, best_r

def _high_precision_polish(centers, time_budget):
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    # Phase 1: Progressive sharpening
    for beta in [500, 1000, 2000, 4000, 8000, 12000]:
        if time.time() - start > time_budget: break
        remaining = time_budget - (time.time() - start)
        if remaining < 4: break
        lb = min(14.0, remaining * 0.85)
        trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lb)
        trial_r = _compute_radius(trial)
        if trial_r > best_r: best_r = trial_r; best = trial
    # Phase 2: Soft-min
    if time.time() - start < time_budget - 8:
        for beta in [200, 500, 1000, 2000, 4000]:
            if time.time() - start > time_budget - 4: break
            remaining = time_budget - (time.time() - start)
            if remaining < 4: break
            lb = min(14.0, remaining * 0.85)
            trial = _lbfgs_soft_min_polish(best.flatten(), beta, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    # Phase 3: Push
    if time.time() - start < time_budget - 4:
        for pf in [1.001, 1.003, 1.005, 1.008, 1.012, 1.018]:
            if time.time() - start > time_budget - 3: break
            rp = best_r * pf
            if rp > 0.08: break
            remaining = time_budget - (time.time() - start)
            if remaining < 3: break
            lb = min(12.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), rp, 4000.0, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    # Phase 4: Ultra-tight
    if time.time() - start < time_budget - 2:
        for beta in [8000, 15000, 25000]:
            if time.time() - start > time_budget - 1.5: break
            remaining = time_budget - (time.time() - start)
            if remaining < 2: break
            lb = min(15.0, remaining * 0.9)
            trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    # Phase 5: SLSQP
    if time.time() - start < time_budget - 3:
        remaining = time_budget - (time.time() - start)
        if remaining > 5:
            trial = _slsqp_polish(best, remaining * 0.6)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    # Phase 6: Coordinate descent
    if time.time() - start < time_budget - 1:
        remaining = time_budget - (time.time() - start)
        if remaining > 3:
            trial, trial_r = _coordinate_descent_polish(best, remaining * 0.8)
            if trial_r > best_r: best_r = trial_r; best = trial
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(777)
    
    inits = [_initial_grid(), _hex_init(), _deformation_init()]
    rng_init = np.random.default_rng(999)
    inits.append(rng_init.uniform(-0.45, 0.45, (N, 2)))
    rng_init2 = np.random.default_rng(4242)
    inits.append(rng_init2.uniform(-0.45, 0.45, (N, 2)))
    
    best_centers = None; best_radius = 0
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r; best_centers = init.copy()
    if best_centers is None:
        best_centers = _initial_grid(); best_radius = _compute_radius(best_centers)
    
    # SA
    sa_total = min(35.0, (deadline - time.time()) * 0.30)
    if sa_total > 8:
        sa1 = min(20.0, sa_total * 0.55)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1)
        if time.time() < deadline - 35:
            rng2 = np.random.default_rng(1234)
            pert = best_centers + rng2.normal(0, 0.008, (N, 2))
            pert = np.clip(pert, -0.49, 0.49)
            sa2 = min(15.0, (deadline - time.time()) - 40)
            if sa2 > 4:
                c2, r2 = _simulated_annealing(pert, rng2, sa2)
                if r2 > best_radius:
                    best_radius = r2; best_centers = c2
    
    # Refine
    if time.time() < deadline - 25:
        rb = min(45.0, deadline - time.time() - 12)
        best_centers, best_radius = _refine_radius(best_centers, rb)
    
    # Polish
    if time.time() < deadline - 6:
        pb = min(30.0, deadline - time.time() - 2)
        best_centers, best_radius = _high_precision_polish(best_centers, pb)
    
    # Final
    if time.time() < deadline - 2:
        fb = min(15.0, deadline - time.time() - 1)
        if fb > 3:
            cr = _compute_radius(best_centers)
            trial = _lbfgs_feasibility_polish(best_centers.flatten(), cr, 20000.0, fb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r; best_centers = trial
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    return best_centers
