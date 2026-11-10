# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 8
_S = 1.0 / 8.0
_K = np.arange(_N)
_DEADLINE = 630.0
_START = time.time()


def _softmax(x, beta):
    m = np.max(x)
    e = np.exp(beta * (x - m))
    return m + np.log(np.sum(e)) / beta


def _softmin(x, beta):
    return -_softmax(-x, beta)


def _union_len_grad(a, beta):
    """Smooth union length of intervals [a_k+k/N, a_k+k/N+1/N] and its gradient."""
    a = np.asarray(a, dtype=np.float64)
    starts = a + _K * _S
    ends = starts + _S
    lo = np.min(starts)
    hi = np.max(ends)
    L = _softmax(ends, beta) - _softmin(starts, beta)
    eE = np.exp(beta * (ends - hi))
    eE = eE / np.sum(eE)
    eS = np.exp(-beta * (starts - lo))
    eS = eS / np.sum(eS)
    g = eE - eS
    return L, g


def _exact_len_grad(a):
    """Exact union length of [a_k+k/N, a_k+k/N+1/N] and exact subgradient (tie-averaged)."""
    a = np.asarray(a, dtype=np.float64)
    s = a + _K * _S
    e = s + _S
    order = np.argsort(s, kind="stable")
    sS = s[order]
    eS = e[order]
    cur = sS[0]
    L = 0.0
    for k in range(_N):
        if eS[k] > cur:
            L += eS[k] - cur
            cur = eS[k]
    # subgradient: for each interval, +1 if it is the unique global max end, -1 if unique global min start
    g = np.zeros(_N, dtype=np.float64)
    g[np.argmax(e)] = 1.0
    g[np.argmin(s)] = -1.0
    # tie averaging: if multiple share the extreme, split the unit mass equally
    emax = np.max(e)
    mask = np.abs(e - emax) < 1e-12
    if np.count_nonzero(mask) > 1:
        g[mask] = 0.0
        g[mask] = 1.0 / np.count_nonzero(mask)
    smin = np.min(s)
    mask2 = np.abs(s - smin) < 1e-12
    if np.count_nonzero(mask2) > 1:
        g[mask2] = 0.0
        g[mask2] = -1.0 / np.count_nonzero(mask2)
    return L, g


def _objective(p):
    Lx, gx = _union_len_grad(p[:_N], 22.0)
    Ly, gy = _union_len_grad(p[_N:], 22.0)
    return Lx * Ly


def _objective_grad(p):
    Lx, gx = _union_len_grad(p[:_N], 22.0)
    Ly, gy = _union_len_grad(p[_N:], 22.0)
    return np.concatenate([Ly * gx, Lx * gy])


def _objective_exact(p):
    Lx, _ = _exact_len_grad(p[:_N])
    Ly, _ = _exact_len_grad(p[_N:])
    return Lx * Ly


def _objective_exact_grad(p):
    Lx, gx = _exact_len_grad(p[:_N])
    Ly, gy = _exact_len_grad(p[_N:])
    return np.concatenate([Ly * gx, Lx * gy])


def _keich():
    pos = int(np.log2(_N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(_N, dtype=np.float64)
    for idx in range(_N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _seed_positions():
    keich = _keich()
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _valid(positions):
    if not (np.isfinite(positions).all()):
        return False
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (np.arange(_N) // _N + 1) * _S)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (np.arange(_N) % _N + 1) * _S)
    return (X1 - X0) > 1e-9 and (Y1 - Y0) > 1e-9


def _mc_volume(positions, n=200000):
    """Fast Monte-Carlo sanity check of the true union volume (seeded)."""
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (np.arange(_N) // _N + 1) * _S)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (np.arange(_N) % _N + 1) * _S)
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return None
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n)
    b = rng.uniform(Y0, Y1, n)
    c = rng.uniform(0.0, 1.0, n)
    ii = (np.arange(_N) // _N).astype(np.float64)
    jj = (np.arange(_N) % _N).astype(np.float64)
    xs = positions[:, 0][:, None] + c[None, :] * ii[None, :]
    xe = xs + _S
    ys = positions[:, 1][:, None] + c[None, :] * jj[None, :]
    ye = ys + _S
    inside = ((a[None, :] >= xs) & (a[None, :] <= xe) &
              (b[None, :] >= ys) & (b[None, :] <= ye)).any(axis=0)
    return float(np.mean(inside)) * (X1 - X0) * (Y1 - Y0)


def kakeya_needle_3d8() -> np.ndarray:
    global _START
    t0 = time.time()

    def time_left():
        return _DEADLINE - (time.time() - _START)

    keich = _keich()
    seed_p = np.concatenate([keich, keich])
    best = seed_p.copy()
    best_val = _objective_exact(best)
    # also track the seed-program positions as a fallback
    seed_positions = _seed_positions()
    if _valid(seed_positions):
        best = seed_positions.copy()
        best_val = _objective_exact(best)

    rng = np.random.default_rng(12345)

    # ---- Phase 1: multi-start L-BFGS-B on the smooth surrogate ----
    n_starts = 10
    for s in range(n_starts):
        if time_left() < 60.0:
            break
        if s == 0:
            p0 = seed_p.copy()
        else:
            p0 = seed_p + rng.normal(0.0, 0.06, size=2 * _N)
        try:
            res = minimize(_objective, p0, jac=_objective_grad, method="L-BFGS-B",
                           options={"maxiter": 2000, "maxfun": 100000,
                                    "ftol": 1e-15, "gtol": 1e-12, "iprint": 0})
            if res.success and np.isfinite(res.fun):
                cand = np.asarray(res.x)
                val = _objective_exact(cand)
                if val < best_val:
                    best = cand
                    best_val = val
        except Exception:
            pass

    # ---- Phase 2: high-precision polish on the EXACT objective ----
    for _ in range(30):
        if time_left() < 30.0:
            break
        try:
            res = minimize(_objective_exact, best, jac=_objective_exact_grad,
                           method="L-BFGS-B",
                           options={"maxiter": 2000, "maxfun": 200000,
                                    "ftol": 1e-18, "gtol": 1e-14, "iprint": 0})
            if np.isfinite(res.fun):
                cand = np.asarray(res.x)
                val = _objective_exact(cand)
                if val < best_val:
                    best = cand
                    best_val = val
                elif res.fun >= best_val - 1e-16:
                    break
        except Exception:
            break

    # ---- Build output positions from best 1-D coordinates ----
    bx = best[:_N]
    by = best[_N:]
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    positions = np.stack([bx[i_grid.flatten()], by[j_grid.flatten()]], axis=1)

    if not _valid(positions):
        positions = _seed_positions()

    return positions


# EVOLVE-BLOCK-END
