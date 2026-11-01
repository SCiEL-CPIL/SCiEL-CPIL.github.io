"""Pack 50 equal circles in the unit circle. Lattice init + soft-min annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150s)
_T0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _T0)


# precomputed upper-triangle index pairs
_IU = np.triu_indices(N, 1)
_MPAIR = _IU[0].shape[0]


def _hex_init(rng=None):
    """Hexagonal lattice cut to the unit disk, N closest points to the origin."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    for _ in range(60):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows, dtype=float)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            order = np.argsort(rad, kind="stable")
            out = pts[order[:N]].copy()
            if rng is not None:
                out = out + rng.normal(0.0, spacing * 0.05, size=out.shape)
            return out
        spacing *= 0.97
    # fallback: ring packing
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([0.6 * np.cos(theta), 0.6 * np.sin(theta)])


def _softmin_and_grad(x, s):
    """Return (softmin value, gradient of softmin wrt x) for the max-min terms."""
    c = x.reshape(N, 2)
    diff = c[:, None, :] - c[None, :, :]            # (N,N,2)
    d = np.sqrt((diff * diff).sum(axis=2))          # (N,N)
    d = np.maximum(d, 1e-12)
    np.fill_diagonal(d, 1.0)
    i, j = _IU
    dij = d[i, j]
    vec = diff[i, j]                                # c_i - c_j
    dpair = dij / 2.0
    normc = np.linalg.norm(c, axis=1)
    bound = 1.0 - normc
    terms = np.concatenate([dpair, bound])
    neg = -s * terms
    m = neg.max()
    e = np.exp(neg - m)
    se = e.sum()
    softmin = -(m + np.log(se)) / s
    w = e / se
    wpair, wbound = w[:_MPAIR], w[_MPAIR:]
    g = np.zeros((N, 2))
    coef = wpair / (2.0 * dij)
    contrib = coef[:, None] * vec
    np.add.at(g, i, contrib)
    np.add.at(g, j, -contrib)
    nc = np.maximum(normc, 1e-12)
    g += -wbound[:, None] * (c / nc[:, None])
    return softmin, g


def _fun(x, s):
    return -_softmin_and_grad(x, s)[0]


def _jac(x, s):
    return -_softmin_and_grad(x, s)[1].ravel()


def _true_r(x):
    c = x.reshape(N, 2)
    if not np.all(np.isfinite(c)):
        return -np.inf, False
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_bound = 1.0 - np.linalg.norm(c, axis=1).min()
    r = min(min_pair / 2.0, min_bound)
    valid = (r > 0.0) and (min_pair > 0.0) and (min_bound > 0.0)
    return r, valid


def _true_min_fun(x):
    r, _ = _true_r(x)
    return -r


def _optimize_from(x0):
    """Anneal soft-min sharpness; return best (r, x) seen."""
    best_r, best_x = _true_r(x0)
    best_x = x0.copy()
    schedule = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    x = x0.copy()
    for s in schedule:
        if _time_left() < 25.0:
            break
        try:
            res = minimize(_fun, x, jac=_jac, args=(s,), method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
            x = res.x
        except Exception:
            break
        r, valid = _true_r(x)
        if valid and r > best_r:
            best_r, best_x = r, x.copy()
    return best_r, best_x


def _polish(x):
    """Final high-s sharpness + Nelder-Mead on the true minimum."""
    best_r, best_x = _true_r(x)
    best_x = x.copy()
    for s in [8192, 16384]:
        if _time_left() < 15.0:
            break
        try:
            res = minimize(_fun, best_x, jac=_jac, args=(s,), method="L-BFGS-B",
                           options={"maxiter": 150, "ftol": 1e-13, "gtol": 1e-9})
            r, valid = _true_r(res.x)
            if valid and r > best_r:
                best_r, best_x = r, res.x.copy()
        except Exception:
            pass
    if _time_left() > 12.0:
        try:
            res = minimize(_true_min_fun, best_x, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-12})
            r, valid = _true_r(res.x)
            if valid and r > best_r:
                best_r, best_x = r, res.x.copy()
        except Exception:
            pass
    return best_r, best_x


def construct():
    global _T0
    _T0 = time.time()
    rng = np.random.default_rng(12345)
    fallback = _hex_init()
    fb_r, fb_valid = _true_r(fallback)
    best_r = fb_r if fb_valid else -np.inf
    best_x = fallback.copy()

    inits = [fallback, _hex_init(rng)]
    for x0 in inits:
        if _time_left() < 40.0:
            break
        r, x = _optimize_from(x0)
        if r > best_r:
            best_r, best_x = r, x
        if _time_left() < 30.0:
            break

    if _time_left() > 15.0:
        r, x = _polish(best_x)
        if r > best_r:
            best_r, best_x = r, x

    out = np.asarray(best_x, dtype=float).reshape(N, 2)
    # final safety: ensure validity, else fall back
    r, valid = _true_r(out)
    if not valid:
        out = fallback
    return out
