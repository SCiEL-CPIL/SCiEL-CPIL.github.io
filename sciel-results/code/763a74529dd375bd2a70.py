"""Circle packing: 50 equal circles in unit square [-1/2,1/2]^2.
Design: vectorized soft-min L-BFGS-B (no per-call constraint overhead) for the
main search, with a final SLSQP contact-graph polish.  Soft-min of the 1250
pairwise distances + 100 wall distances gives a smooth proxy for the true
min-radius; annealing the inverse-temperature beta drives it toward the exact
min.  A hard penalty keeps circles inside the square.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5

# ---- precompute pair indices (flat, for vectorized ops) ----
_I, _J = np.triu_indices(N, 1)
_PAIRS = [(i, j) for i, j in zip(_I, _J)]

# ---- soft-min objective (vectorized, no per-call closure overhead) ----
def _softmin_obj(p, beta, mu):
    """Soft-min of pairwise distances/2 and wall distances, minus hard
    boundary penalty.  Returns (value, grad) for L-BFGS-B."""
    pos = p.reshape(N, 2)
    # pairwise distances (vectorized)
    dx = pos[_I, 0] - pos[_J, 0]
    dy = pos[_I, 1] - pos[_J, 1]
    d = np.sqrt(dx * dx + dy * dy + 1e-30)
    pd = d / 2.0                       # pairwise min-radius candidates
    # wall distances
    wx = HALF - np.abs(pos[:, 0])
    wy = HALF - np.abs(pos[:, 1])
    wall = np.minimum(wx, wy)          # wall min-radius candidates

    # soft-min over all candidates
    all_d = np.concatenate([pd, wall])
    # clip for numerical stability in exp
    all_d_c = np.clip(all_d, 1e-12, None)
    exp_b = np.exp(-beta * all_d_c)
    Z = exp_b.sum()
    sm = -np.log(Z) / beta             # soft-min value
    # d(sm)/d(all_d_k) = exp(-beta*d_k)/Z   (positive: increasing d_k increases sm)
    w = exp_b / Z

    # gradient w.r.t. positions
    g = np.zeros((N, 2))
    # pairwise part: w[k] * d(pd_k)/d(pos)
    # pd_k = d_k/2,  d(d_k)/d(pos_i) = (dx_k, dy_k)/d_k  for i=_I[k]
    # d(d_k)/d(pos_j) = -(dx_k, dy_k)/d_k  for j=_J[k]
    inv_d = 1.0 / d
    gx_i = w[:len(_I)] * 0.5 * dx * inv_d
    gy_i = w[:len(_I)] * 0.5 * dy * inv_d
    np.add.at(g[:, 0], _I,  gx_i)
    np.add.at(g[:, 1], _I,  gy_i)
    np.add.at(g[:, 0], _J, -gx_i)
    np.add.at(g[:, 1], _J, -gy_i)
    # wall part: w[1250+k] * d(wall_k)/d(pos_k)
    ww = w[len(_I):]
    # d(wx)/dx = -sign(x), d(wy)/dy = -sign(y)
    g[:, 0] -= ww * np.sign(pos[:, 0])
    g[:, 1] -= ww * np.sign(pos[:, 1])

    # hard penalty for going outside
    pen = 0.0
    pen_g = np.zeros((N, 2))
    # x outside
    o = pos[:, 0] - HALF
    m = o > 0
    if np.any(m):
        pen += mu * 0.5 * (o[m] ** 2).sum()
        pen_g[m, 0] += mu * o[m]
    o = -HALF - pos[:, 0]
    m = o > 0
    if np.any(m):
        pen += mu * 0.5 * (o[m] ** 2).sum()
        pen_g[m, 0] -= mu * o[m]
    # y outside
    o = pos[:, 1] - HALF
    m = o > 0
    if np.any(m):
        pen += mu * 0.5 * (o[m] ** 2).sum()
        pen_g[m, 1] += mu * o[m]
    o = -HALF - pos[:, 1]
    m = o > 0
    if np.any(m):
        pen += mu * 0.5 * (o[m] ** 2).sum()
        pen_g[m, 1] -= mu * o[m]

    val = -sm + pen
    grad = (-g + pen_g).ravel()
    return val, grad

def _softmin_fun(p, beta, mu):
    return _softmin_obj(p, beta, mu)[0]

def _softmin_grad(p, beta, mu):
    return _softmin_obj(p, beta, mu)[1]

# ---- true (exact) min-radius of a configuration ----
def _radius(coords):
    r = (HALF - np.abs(coords)).min()
    dx = coords[_I, 0] - coords[_J, 0]
    dy = coords[_I, 1] - coords[_J, 1]
    r = min(r, np.sqrt(dx * dx + dy * dy).min() / 2.0)
    return float(r)

# ---- contact-graph SLSQP polish (only active constraints) ----
def _slsqp_polish(pos, r0, maxiter=200):
    """SLSQP on the contact graph at radius r0.  Returns improved positions."""
    # find contacts: pairs with dist < r0*2 + tol, and walls within tol
    tol = r0 * 0.05 + 1e-6
    dx = pos[_I, 0] - pos[_J, 0]
    dy = pos[_I, 1] - pos[_J, 1]
    d = np.sqrt(dx * dx + dy * dy)
    pair_active = d < r0 * 2.0 + tol
    # wall active
    wx = HALF - np.abs(pos[:, 0])
    wy = HALF - np.abs(pos[:, 1])
    wall_active = np.minimum(wx, wy) < r0 + tol

    cons = []
    # pairwise active
    for k in np.where(pair_active)[0]:
        i, j = _I[k], _J[k]
        ii, jj = 2 * i, 2 * j
        def fun(x, ii=ii, jj=jj):
            return np.hypot(x[ii] - x[jj], x[ii + 1] - x[jj + 1]) - 2.0 * x[-1]
        def jac(x, ii=ii, jj=jj):
            g = np.zeros_like(x)
            dx_ = x[ii] - x[jj]; dy_ = x[ii + 1] - x[jj + 1]
            nrm = np.hypot(dx_, dy_)
            if nrm < 1e-12:
                nrm = 1e-12
            g[ii] = dx_ / nrm; g[ii + 1] = dy_ / nrm
            g[jj] = -dx_ / nrm; g[jj + 1] = -dy_ / nrm
            g[-1] = -2.0
            return g
        cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
    # wall active
    for i in range(N):
        for k in range(2):
            idx = 2 * i + k
            if wall_active[i]:
                def fun(x, idx=idx):
                    return (HALF - abs(x[idx])) - x[-1]
                def jac(x, idx=idx):
                    g = np.zeros_like(x)
                    g[idx] = -1.0 if x[idx] >= 0 else 1.0
                    g[-1] = -1.0
                    return g
                cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})

    if not cons:
        return pos

    x0 = np.concatenate([pos.ravel(), [r0]])
    bounds = [(-HALF, HALF)] * (2 * N) + [(0.0, HALF)]
    def obj(x): return -x[-1]
    def objg(x):
        g = np.zeros_like(x); g[-1] = -1.0; return g
    try:
        res = minimize(obj, x0, method='SLSQP', jac=objg, bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success or res.fun < obj(x0):
            return res.x[:2 * N].reshape(N, 2)
    except Exception:
        pass
    return pos

# ---- initial guesses ----
def _grid_initial():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_initial():
    dx = 0.14
    dy = dx * np.sqrt(3.0) / 2.0
    pts = []
    row = 0
    y = -HALF + dx / 2.0
    while len(pts) < N and row < 200:
        off = (dx / 2.0) if (row % 2) else 0.0
        x = -HALF + dx / 2.0 + off
        while x <= HALF and len(pts) < N:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(pts[:N])

def construct():
    t0 = time.time()
    BUDGET = 135.0          # safety margin before 150s kill
    best = None
    best_r = -1.0

    def consider(coords):
        nonlocal best, best_r
        coords = np.clip(coords, -HALF, HALF)
        if not np.all(np.isfinite(coords)):
            return
        r = _radius(coords)
        if r > best_r:
            best_r = r
            best = coords.copy()

    def softmin_run(pos0, beta_schedule, mu=1000.0):
        """L-BFGS-B soft-min optimizer with beta annealing.  Returns final pos."""
        p = pos0.ravel().copy()
        for beta in beta_schedule:
            if time.time() - t0 > BUDGET - 5.0:
                break
            try:
                res = minimize(
                    lambda pp: _softmin_fun(pp, beta, mu),
                    p,
                    jac=lambda pp: _softmin_grad(pp, beta, mu),
                    method='L-BFGS-B',
                    bounds=[(-HALF * 1.5, HALF * 1.5)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-14,
                             'gtol': 1e-10, 'maxls': 50},
                )
                p = res.x
            except Exception:
                pass
        return p.reshape(N, 2)

    # candidate starts: dense hex, square grid, and random perturbations
    rng = np.random.default_rng(0)
    starts = [
        _hex_initial(),
        _grid_initial(),
        _hex_initial() + 0.015 * rng.standard_normal((N, 2)),
        _grid_initial() + 0.015 * rng.standard_normal((N, 2)),
        _hex_initial() + 0.03 * rng.standard_normal((N, 2)),
    ]

    # beta schedule: anneal from low to high for sharper min-tracking
    beta_sched = [20.0, 50.0, 100.0, 200.0, 400.0]

    for init in starts:
        if time.time() - t0 > BUDGET - 20.0:
            break
        pos = softmin_run(init, beta_sched)
        consider(pos)

    # contact-graph SLSQP polish on the best configuration
    if best is not None and time.time() - t0 < BUDGET - 8.0:
        try:
            pos = _slsqp_polish(best, best_r, maxiter=300)
            consider(pos)
        except Exception:
            pass

    # second soft-min pass from polished best, with even higher beta
    if best is not None and time.time() - t0 < BUDGET - 5.0:
        pos = softmin_run(best, [400.0, 800.0, 1500.0], mu=5000.0)
        consider(pos)

    # final SLSQP polish
    if best is not None and time.time() - t0 < BUDGET - 3.0:
        try:
            pos = _slsqp_polish(best, best_r, maxiter=200)
            consider(pos)
        except Exception:
            pass

    if best is None:
        best = _grid_initial()
    return best
