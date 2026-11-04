"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
T_LIMIT = 140.0   # safety margin under the 150 s wall-clock kill
T0 = time.time()

def _elapsed():
    return time.time() - T0

def _remaining():
    return max(0.0, T_LIMIT - _elapsed())

# ----------------------------------------------------------------------
# Softplus-based smooth objective + analytic gradient
# ----------------------------------------------------------------------
def _softmax(x, beta):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m))) / beta

def _obj_and_grad(z, beta):
    n = len(z)
    pos = z[:2 * n]
    rad = z[2 * n:]
    x = pos[0::2]
    y = pos[1::2]
    r = rad

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-16)
    np.fill_diagonal(d, 0.0)
    s = r[:, None] + r[None, :]

    # Smooth overlap penalty: softplus(beta*(s - d))
    u = beta * (s - d)
    uu = np.where(u > 0, u, 0.0)
    sp = np.log1p(np.exp(-uu)) / beta + (u > 0) * (u / beta) * 0 + np.where(u > 0, u / beta, 0.0)
    # simpler: softplus(u)/beta
    sp = (np.where(u > 0, u, 0.0) + np.log1p(np.exp(-np.abs(u)))) / beta

    # Boundary penalty: softplus(beta*(r - min(x,1-x))) etc.
    lo_x = r - x
    hi_x = r - (1.0 - x)
    lo_y = r - y
    hi_y = r - (1.0 - y)

    def _sp1(v):
        v = np.asarray(v)
        return (np.where(v > 0, v, 0.0) + np.log1p(np.exp(-np.abs(v)))) / beta

    pen = (_sp1(lo_x).sum() + _sp1(hi_x).sum() + _sp1(lo_y).sum() + _sp1(hi_y).sum())

    obj = -r.sum() + pen

    # Gradient
    grad = np.zeros(n)

    # pairwise contribution to positions and radii
    # d/d(x_i) sp(beta*(s_ij - d_ij)) = -beta * sp' * (dx_ij/dx_i)
    # sp'(u) = sigmoid(u)
    sig = 1.0 / (1.0 + np.exp(-u))
    # gradient wrt x_i: sum_j -beta*sig_ij * ( (x_i - x_j)/d_ij )  for i!=j
    # build matrix w_ij = beta * sig_ij / d_ij
    W = np.zeros((n, n))
    np.fill_diagonal(W, 0.0)
    W = beta * sig / np.where(d > 1e-12, d, 1e-12)
    np.fill_diagonal(W, 0.0)

    g_x = - (W * dx).sum(axis=1)   # dx[i,j] = x_i - x_j
    g_y = - (W * dy).sum(axis=1)
    g_r = W.sum(axis=1)            # d/d r_i of sp(beta*(s_ij-d_ij)) = beta*sig/beta = sig ; sum over j
    # note: s_ij = r_i + r_j, d/d r_i = beta*sig_ij/beta = sig_ij ; summed -> sig.sum(axis=1)
    g_r = sig.sum(axis=1)

    # boundary gradients
    def _spg1(v):
        v = np.asarray(v)
        return 1.0 / (1.0 + np.exp(-v / 1.0)) if False else None
    # softplus(v)/beta derivative wrt v = sigmoid(v)
    def _sig(v):
        v = np.asarray(v)
        return 1.0 / (1.0 + np.exp(-v))

    # lo_x = r - x ; d/dx = -sig(lo_x), d/dr = +sig(lo_x)
    sx_lo = _sig(lo_x)
    sx_hi = _sig(hi_x)
    sy_lo = _sig(lo_y)
    sy_hi = _sig(hi_y)

    g_x = g_x - sx_lo + sx_hi
    g_y = g_y - sy_lo + sy_hi
    g_r = g_r + sx_lo + sx_hi + sy_lo + sy_hi

    grad[:n] = g_x
    grad[n:2 * n] = g_y
    grad[2 * n:] = g_r
    return obj, grad

def _valid(pos, rad, tol=1e-6):
    x = pos[0::2]; y = pos[1::2]; r = rad
    if np.any(r < -tol):
        return False
    r = np.clip(r, 0, None)
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 0.0)
    if np.any(d < (r[:, None] + r[None, :]) - tol):
        return False
    return True

def _value(pos, rad):
    if not _valid(pos, rad):
        return -np.inf
    return float(np.sum(rad))

# ----------------------------------------------------------------------
# Analytic starting layouts
# ----------------------------------------------------------------------
def _grid_layout():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    z = np.zeros(3 * N)
    for k, (x, y, r) in enumerate(cells):
        z[2 * k] = x; z[2 * k + 1] = y; z[2 * N + k] = r
    return z

def _random_layout(rng, spread=0.12):
    z = np.zeros(3 * N)
    # random positions in interior
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.02, 0.09, N)
    z[:N] = x; z[N:2 * N] = y; z[2 * N:] = r
    return z

def _hex_layout(rng, jitter=0.0):
    # approximate hexagonal packing start
    pts = []
    rows = 0
    # target ~ N points
    spacing = 0.16
    for j in range(8):
        for i in range(8):
            xx = 0.08 + i * spacing + (0.08 if j % 2 else 0.0)
            yy = 0.08 + j * spacing * 0.866
            if xx < 0.92 and yy < 0.92:
                pts.append((xx, yy))
    rng.shuffle(pts)
    pts = pts[:N]
    z = np.zeros(3 * N)
    for k, (x, y) in enumerate(pts):
        z[2 * k] = x; z[2 * k + 1] = y; z[2 * N + k] = 0.07
    if jitter:
        z[:2 * N] += rng.uniform(-jitter, jitter, 2 * N)
    return z

# ----------------------------------------------------------------------
# L-BFGS-B refinement with annealed beta
# ----------------------------------------------------------------------
def _refine(z0, beta_start=20.0, beta_end=400.0, iters=6, max_iter=400, time_budget=20.0):
    z = z0.copy()
    betas = np.linspace(np.log(beta_start), np.log(beta_end), iters)
    for it in range(iters):
        b = float(np.exp(betas[it]))
        t0 = time.time()
        res = minimize(_obj_and_grad, z, method='L-BFGS-B',
                       jac=True, options={'maxiter': max_iter, 'maxfun': max_iter * 4,
                                          'gtol': 1e-8, 'ftol': 1e-12})
        if res.x is not None:
            z = res.x
        if time.time() - t0 > time_budget:
            break
        if _remaining() < 5:
            break
    return z

def _clip(z):
    z = z.copy()
    z[:N] = np.clip(z[:N], 0.0, 1.0)
    z[N:2 * N] = np.clip(z[N:2 * N], 0.0, 1.0)
    z[2 * N:] = np.clip(z[2 * N:], 0.0, 0.5)
    return z

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def construct():
    rng = np.random.default_rng(12345)

    starts = [_grid_layout(), _hex_layout(rng, jitter=0.01)]
    for _ in range(6):
        starts.append(_random_layout(rng))

    best_z = None
    best_val = -np.inf

    # First, refine each start to get good candidates
    for z0 in starts:
        if _remaining() < 8:
            break
        z = _refine(z0, beta_start=20.0, beta_end=300.0, iters=5,
                    max_iter=300, time_budget=12.0)
        z = _clip(z)
        v = _value(z[:2 * N], z[2 * N:])
        if v > best_val:
            best_val = v
            best_z = z.copy()

    # Second stage: local perturbations around the best, higher beta
    if best_z is not None and _remaining() > 15:
        base = best_z.copy()
        for _ in range(40):
            if _remaining() < 8:
                break
            # small perturbation
            z0 = base + rng.normal(0, 0.01, 3 * N)
            z0 = _clip(z0)
            z = _refine(z0, beta_start=80.0, beta_end=600.0, iters=4,
                        max_iter=250, time_budget=8.0)
            z = _clip(z)
            v = _value(z[:2 * N], z[2 * N:])
            if v > best_val:
                best_val = v
                best_z = z.copy()
                base = best_z.copy()

    # Final high-precision polish on the best
    if best_z is not None and _remaining() > 10:
        z = _refine(best_z, beta_start=300.0, beta_end=800.0, iters=4,
                    max_iter=400, time_budget=15.0)
        z = _clip(z)
        v = _value(z[:2 * N], z[2 * N:])
        if v > best_val:
            best_val = v
            best_z = z.copy()

    if best_z is None:
        best_z = _grid_layout()

    # Build output array (x, y, radius)
    out = np.zeros((N, 3))
    out[:, 0] = best_z[:N]
    out[:, 1] = best_z[N:2 * N]
    out[:, 2] = best_z[2 * N:]
    return out
