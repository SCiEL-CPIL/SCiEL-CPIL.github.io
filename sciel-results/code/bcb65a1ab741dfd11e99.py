"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# ---------- global deadline guard ----------
_DEADLINE = time.time() + 115.0   # safety margin well under the 150s kill limit


def _time_left():
    return _DEADLINE - time.time()


def _soft_max(v, beta):
    """Smooth max of a vector (log-sum-exp), with exact fallback for large beta."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return 0.0
    m = float(np.max(v))
    if m < -1e-12:
        return 0.0
    if beta > 2000.0:
        return m
    z = np.exp(beta * (v - m))
    return m + np.log(np.sum(z)) / beta


def _objective_and_grad(z, beta, gmax, gwall):
    """
    Minimize  -sum(r) + gmax * sum_pairs soft(d_ij - (r_i+r_j)) + gwall * soft(wall violations).
    z layout: [x(26), y(26), r(26)].
    """
    n = z.size // 3
    x = z[0:n]
    y = z[n:2 * n]
    r = z[2 * n:3 * n]

    # pair distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 0.0)
    rad = r[:, None] + r[None, :]

    # ---- value ----
    viol = d - rad
    m = float(np.max(viol))
    if m < -1e-12:
        smax = 0.0
    elif beta > 2000.0:
        smax = m
    else:
        smax = m + np.log(np.sum(np.exp(beta * (viol - m)))) / beta

    wall = np.concatenate([
        r - x, r - y, r - (1.0 - x), r - (1.0 - y)
    ])
    mw = float(np.max(wall))
    if mw < -1e-12:
        swall = 0.0
    elif beta > 2000.0:
        swall = mw
    else:
        swall = mw + np.log(np.sum(np.exp(beta * (wall - mw)))) / beta

    value = -np.sum(r) + gmax * smax + gwall * swall

    # ---- gradient ----
    if m < -1e-12:
        w = np.zeros_like(viol)
    elif beta > 2000.0:
        w = np.zeros_like(viol)
        w[viol == m] = 1.0
        w /= w.sum()
    else:
        e = np.exp(beta * (viol - m))
        w = e / np.sum(e)
    gw = w * gmax

    if mw < -1e-12:
        wwall = np.zeros_like(wall)
    elif beta > 2000.0:
        wwall = np.zeros_like(wall)
        wwall[wall == mw] = 1.0
        wwall /= wwall.sum()
    else:
        ew = np.exp(beta * (wall - mw))
        wwall = ew / np.sum(ew)
    gww = wwall * gwall

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # pair contributions
    for i in range(n):
        for j in range(i + 1, n):
            c = gw[i, j]
            if c == 0.0:
                continue
            dij = d[i, j]
            if dij > 1e-10:
                ux = dx[i, j] / dij
                uy = dy[i, j] / dij
            else:
                ux = 0.5
                uy = 0.5
            gx[i] += c * ux
            gx[j] -= c * ux
            gy[i] += c * uy
            gy[j] -= c * uy
            gr[i] += c
            gr[j] += c

    # wall contributions: wall = [r-x, r-y, r-(1-x), r-(1-y)] per i
    for i in range(n):
        a = gww[4 * i]      # r - x
        b = gww[4 * i + 1]  # r - y
        c = gww[4 * i + 2]  # r - (1-x)
        d_ = gww[4 * i + 3] # r - (1-y)
        gx[i] += -a + c
        gy[i] += -b + d_
        gr[i] += a + b + c + d_

    grad = np.concatenate([gx, gy, gr])
    return value, grad


def _initial_guess(rng, mode):
    n = N
    if mode == 0:  # near-uniform grid
        side = int(np.ceil(np.sqrt(n)))
        idx = 0
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        r = np.full(n, 0.5 / side)
    else:  # random
        x = rng.uniform(0.15, 0.85, n)
        y = rng.uniform(0.15, 0.85, n)
        r = np.full(n, 0.10)
    return np.concatenate([x, y, r])


def _sum_radii(z):
    return float(z[2 * N:3 * N].sum())


def _validity(z, tol=1e-6):
    n = N
    x = z[0:n]
    y = z[n:2 * n]
    r = z[2 * n:3 * n]
    if np.any(r <= 0):
        return False
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x > 1 + tol) or np.any(y > 1 + tol):
        return False
    if np.any(x < r + tol) or np.any(y < r + tol) or np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 0.0)
    rad = r[:, None] + r[None, :]
    if np.any(d < rad - tol):
        return False
    return True


def construct():
    rng = np.random.default_rng(0xC0FFEE)
    best = None
    best_val = -np.inf

    modes = [0, 1, 1, 1, 0, 1, 1, 0]
    beta_start = 30.0
    beta_end = 3000.0
    gmax = 100.0
    gwall = 100.0

    # annealing schedule
    n_anneal = 6
    betas = np.geomspace(beta_start, beta_end, n_anneal)

    for it in range(40):
        if _time_left() < 6.0:
            break
        mode = modes[it % len(modes)]
        z0 = _initial_guess(rng, mode)
        # small random perturbation of start to diversify
        if mode == 0:
            z0 = z0 + rng.normal(0, 1e-3, z0.size)

        for b in betas:
            if _time_left() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad,
                    z0,
                    args=(b, gmax, gwall),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 1 - 1e-6)] * (2 * N) + [(1e-4, 1.0)] * N,
                    options={'maxiter': 1200, 'maxfun': 12000,
                             'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 40},
                )
            except Exception:
                res = None
            if res is not None and res.x is not None:
                zc = res.x
                if _validity(zc, tol=1e-5):
                    v = _sum_radii(zc)
                    if v > best_val:
                        best_val = v
                        best = zc.copy()
                z0 = zc  # carry forward for next (tighter) beta

        # final high-precision polish on the best-so-far
        if best is not None and _time_left() > 5.0:
            try:
                res = minimize(
                    _objective_and_grad,
                    best,
                    args=(beta_end, gmax, gwall),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 1 - 1e-6)] * (2 * N) + [(1e-4, 1.0)] * N,
                    options={'maxiter': 4000, 'maxfun': 60000,
                             'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 60},
                )
                if res.x is not None and _validity(res.x, tol=1e-6):
                    v = _sum_radii(res.x)
                    if v > best_val:
                        best_val = v
                        best = res.x.copy()
            except Exception:
                pass

    if best is None:
        # fallback: trivial valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])

    x = best[0:N]
    y = best[N:2 * N]
    r = best[2 * N:3 * N]
    return np.column_stack([x, y, r])
