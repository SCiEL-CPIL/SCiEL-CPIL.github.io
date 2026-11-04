"""Force-directed contact-graph circle packing in a perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = 150.0
_TIME0 = time.time()
_TIME_BUDGET = _DEADLINE - 8.0   # safety margin before the 150 s kill


def _remaining():
    return _TIME_BUDGET - (time.time() - _TIME0)


def _logsumexp(v, k):
    vmax = np.max(v)
    if vmax == -np.inf:
        return -np.inf
    return vmax + np.log(np.sum(np.exp(v - vmax))) / k


def _build_objective(w, h, k):
    """Return smooth objective and its gradient for a fixed aspect (w,h)."""
    def obj_g(z):
        n = z.size // 3
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        eps = 1e-12
        rr = np.maximum(r, eps)

        # pairwise squared overlap violations
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        s = rr[:, None] + rr[None, :]
        viol = s * s - d2
        viol = np.maximum(viol, 0.0)
        iu = np.triu_indices(n, k=1)
        vpair = viol[iu]

        # smooth barrier terms
        Lse = _logsumexp(vpair, k)
        Lse = Lse if np.isfinite(Lse) else 0.0
        # boundary penetration (smooth)
        pen = np.concatenate([rr - x, w - x - rr, rr - y, h - y - rr])
        pen = np.maximum(pen, 0.0)
        Lb = _logsumexp(pen, k)
        Lb = Lb if np.isfinite(Lb) else 0.0

        lam = 1e2
        obj = -np.sum(r) + lam * (Lse + Lb)

        # gradients
        g = np.empty_like(z)
        g[0::3] = 0.0
        g[1::3] = 0.0
        g[2::3] = 0.0
        g[2::3] -= 1.0  # d(-sum r)/dr

        if np.max(vpair) > 0.0:
            expv = np.exp(vpair - np.max(vpair))
            wgt = expv / np.sum(expv)  # softmax weights over pairs
            # accumulate into pair matrix
            P = np.zeros((n, n))
            P[iu] = wgt
            # d(viol)/d(x_i) = -2 dx_ij, d/d(r_i)=2 s_ij
            gdx = -2.0 * (dx * P).sum(axis=1)
            gdy = -2.0 * (dy * P).sum(axis=1)
            gdr = 2.0 * (s * P).sum(axis=1)
            g[0::3] += lam * gdx
            g[1::3] += lam * gdy
            g[2::3] += lam * gdr

        if np.max(pen) > 0.0:
            expb = np.exp(pen - np.max(pen))
            wgtb = expb / np.sum(expb)
            # pen terms order: [r-x (n), w-x-r (n), r-y (n), h-y-r (n)]
            w1 = wgtb[0:n]
            w2 = wgtb[n:2 * n]
            w3 = wgtb[2 * n:3 * n]
            w4 = wgtb[3 * n:4 * n]
            g[0::3] += lam * (-w1 - w2)
            g[1::3] += lam * (-w3 - w4)
            g[2::3] += lam * (w1 + w2 + w3 + w4)

        return obj, g
    return obj_g


def _is_valid(x, y, r, w, h, tol=1e-7):
    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)
    if np.any(x < r - tol) or np.any(x > w - r - tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r - tol):
        return False
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    if np.any(d < s - tol):
        return False
    return True


def _polish(x, y, r, w, h, k, time_left):
    """High-precision smooth polish from a good starting point."""
    z = np.concatenate([x, y, r])
    og = _build_objective(w, h, k)
    t0 = time.time()
    res = None
    for kk in [k, k * 3.0, k * 9.0]:
        if time.time() - t0 > max(0.2, time_left * 0.6):
            break
        try:
            res = minimize(og, z, jac=True, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxls': 60,
                                    'ftol': 1e-14, 'gtol': 1e-10})
            z = res.x
        except Exception:
            break
    x = z[0::3]
    y = z[1::3]
    r = np.maximum(z[2::3], 0.0)
    return x, y, r


def construct():
    n = N
    rng = np.random.default_rng(12345)

    # candidate aspect ratios (w + h = 2); explore a spread
    aspects = [2.0 / (1.0 + t) for t in
               [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.5, 1.8]]
    aspects = [a for a in aspects if 0.2 < a < 5.0]

    best = None
    best_val = -np.inf
    k = 30.0
    lam = 1e2

    def seed_grid(w, h):
        # grid-like initial layout (square root of N)
        side = int(np.ceil(np.sqrt(n)))
        xs = np.empty(n)
        ys = np.empty(n)
        for idx in range(n):
            i = idx % side
            j = idx // side
            xs[idx] = (i + 0.5) * w / side
            ys[idx] = (j + 0.5) * h / side
        r0 = 0.5 * min(w / side, h / side) * 0.9
        return xs, ys, np.full(n, r0)

    def seed_random(w, h):
        xs = rng.uniform(0.15, w - 0.15, n)
        ys = rng.uniform(0.15, h - 0.15, n)
        r0 = 0.10 * min(w, h)
        return xs, ys, np.full(n, r0)

    n_restarts = 26
    for ri in range(n_restarts):
        if _remaining() < 3.0:
            break
        w = aspects[ri % len(aspects)]
        h = 2.0 - w
        if ri % 2 == 0:
            xs, ys, r = seed_grid(w, h)
        else:
            xs, ys, r = seed_random(w, h)
        z = np.concatenate([xs, ys, r])
        og = _build_objective(w, h, k)
        try:
            res = minimize(og, z, jac=True, method='L-BFGS-B',
                           options={'maxiter': 250, 'maxls': 40,
                                    'ftol': 1e-12, 'gtol': 1e-8})
            z = res.x
        except Exception:
            continue
        x = z[0::3]
        y = z[1::3]
        r = np.maximum(z[2::3], 0.0)
        if not _is_valid(x, y, r, w, h):
            # try to repair with a heavier barrier quickly
            continue
        # polish
        x, y, r = _polish(x, y, r, w, h, k, _remaining())
        if not _is_valid(x, y, r, w, h):
            continue
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best = (x.copy(), y.copy(), r.copy(), w, h)

    if best is None:
        # fallback: return the seed grid packing (valid, perimeter 4)
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) * w / side, (j + 0.5) * h / side,
                  0.5 * min(w / side, h / side) * 0.9)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:n])

    x, y, r, w, h = best
    out = np.empty((n, 3))
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = r
    return out
