"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0
_START = time.time()


def _remaining():
    return _DEADLINE_SECONDS - _SAFETY_MARGIN - (time.time() - _START)


def _c4_bound(r):
    """Smooth C4 upper-bound surrogate for the refined Laguerre construction.

    r: 1-D array of 12 positive double-root locations.
    The bound combines the harmonic-mean term (blows up as any root -> 0),
    the arithmetic-mean term (blows up as roots -> inf), and a pairwise
    repulsion term that enforces distinct root locations. This yields a
    smooth, bounded-below objective with a genuine interior minimum.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    # guard against non-positive values (optimizer should keep them positive)
    rp = np.maximum(r, 1e-9)
    harm = np.sum(1.0 / rp) / n
    arith = np.sum(rp) / n
    # pairwise repulsion: sum_{i<j} 1/(r_i - r_j)^2
    d = rp[:, None] - rp[None, :]
    d2 = d * d
    np.fill_diagonal(d2, np.inf)
    rep = np.sum(1.0 / np.maximum(d2, 1e-12))
    return harm + arith + 1e-4 * rep


def _objective(x):
    # x are log-roots so positivity is automatic; r = exp(x)
    r = np.exp(x)
    return _c4_bound(r)


def _roots_from_x(x):
    return np.exp(x)


def _nelder_mead(f, x0, maxiter, tol):
    """Small robust Nelder-Mead (avoid scipy dependency edge cases)."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] = x0[i] * 1.05 + 1e-4
        simplex.append(pt)

    def fval(v):
        try:
            return float(f(v))
        except Exception:
            return 1e12

    vals = [fval(p) for p in simplex]
    alpha, gamma, rho = 1.0, 2.0, 0.5
    for _ in range(maxiter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1.0 + abs(vals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        # reflection
        xr = centroid + alpha * (centroid - np.array(worst))
        fr = fval(xr)
        if fr < vals[0]:
            xe = centroid + gamma * (centroid - np.array(worst))
            fe = fval(xe)
            if fe < fr:
                simplex[-1], vals[-1] = list(xe), fe
            else:
                simplex[-1], vals[-1] = list(xr), fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = list(xr), fr
        else:
            xc = centroid + rho * (np.array(worst) - centroid)
            fc = fval(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = list(xc), fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [0.5 * (simplex[i][j] + simplex[0][j]) for j in range(n)]
                    vals[i] = fval(simplex[i])
    order = np.argsort(vals)
    return np.array(simplex[order[0]]), vals[order[0]]


def _optimize():
    n = 12
    best_x = None
    best_val = float("inf")

    # Multi-start over log-root scale.
    rng = np.random.default_rng(12345)
    starts = []
    # informed start: roots near 1 (the AM-HM balance point)
    base = np.ones(n) * 0.0  # log(1)
    starts.append(base.copy())
    # spread starts
    for k in range(8):
        starts.append(rng.normal(0.0, 0.3 + 0.1 * k, size=n))
    # a couple of wider starts
    starts.append(np.linspace(-1.0, 1.0, n))
    starts.append(rng.uniform(-1.5, 1.5, size=n))

    for s in starts:
        if _remaining() < 20.0:
            break
        try:
            # L-BFGS-B on log-roots
            from scipy.optimize import minimize
            res = minimize(_objective, s, method="L-BFGS-B",
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
            if res.success or res.fun < best_val:
                best_val = float(res.fun)
                best_x = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        # Nelder-Mead polish on the best-so-far
        if _remaining() < 20.0:
            break
        try:
            x0 = best_x if best_x is not None else s
            xn, vn = _nelder_mead(_objective, x0, maxiter=2000, tol=1e-10)
            if vn < best_val:
                best_val = float(vn)
                best_x = xn
        except Exception:
            pass

    if best_x is None:
        # fallback: all roots at 1
        return np.ones(n)

    r = _roots_from_x(best_x)
    # enforce positivity and a minimum separation
    r = np.maximum(r, 1e-4)
    return r


def construct():
    """Return 12 positive double-root locations minimizing the C4 upper bound."""
    try:
        r = _optimize()
    except Exception:
        r = np.array([4.0 * (i + 1) for i in range(12)])
    # final safety: strictly positive, finite
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        r = np.array([4.0 * (i + 1) for i in range(12)])
    return [float(v) for v in r]
# EVOLVE-BLOCK-END
