"""Uncertainty inequality: tune Hermite coefficients to minimize the C4 bound.

We keep the seed design (optimize the 3 free coefficients a0,a4,a8; a12 is
fixed so g(0)=0) but replace the fixed prior-state-of-the-art point with a
high-fidelity, robust local optimization that starts *from* that known-good
point. This guarantees we never regress below the seed score and lets us
polish the solution with a smooth (soft-min) epigraph objective.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 200.0  # safety margin before the 210 s kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _hermite_coeffs(a):
    """Return Hermite coefficients [c0, c4, c8, c12]; a12 fixed so g(0)=0."""
    a0, a4, a8 = a
    # g(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12, with g(0)=0.
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=30240.
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 30240.0
    return np.array([a0, a4, a8, a12])


def _poly(c, x):
    """Evaluate the Hermite combination c0*H0+...+c12*H12 at x (vectorized)."""
    c0, c4, c8, c12 = c
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0 * x2 ** 2 - 48.0 * x2 + 12.0
    H8 = 256.0 * x2 ** 4 - 1344.0 * x2 ** 3 + 2688.0 * x2 ** 2 - 1344.0 * x2 + 168.0
    H12 = (4096.0 * x2 ** 6 - 46080.0 * x2 ** 5 + 241920.0 * x2 ** 4
           - 516480.0 * x2 ** 3 + 451584.0 * x2 ** 2 - 145152.0 * x2 + 3024.0)
    return c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12


def _bound(c):
    """Upper bound C4 = sup_x |g(x)| e^{x^2/2} / (||g||_2 * ||g_hat||_2).

    g_hat(x) = g(-x) (Fourier of even function), so ||g_hat||_2 = ||g||_2.
    We maximize the pointwise ratio over a fine grid (soft-min epigraph).
    """
    x = np.linspace(0.0, 9.0, 20001)
    g = _poly(c, x)
    norm2 = np.sqrt(np.sum(g * g) * (x[1] - x[0]))
    if norm2 < 1e-300:
        return np.inf
    ratio = np.abs(g) * np.exp(0.5 * x * x) / (norm2 * norm2)
    # soft-max (log-sum-exp) as a smooth upper bound on max(ratio)
    m = np.max(ratio)
    if not np.isfinite(m) or m <= 0.0:
        return 0.0
    logsum = m + np.log(np.sum(np.exp(ratio - m)))
    return np.exp(logsum)


def _smooth_objective(a, t):
    """Epigraph soft objective: t + (1/k) log sum exp(k*(ratio_i - t))."""
    c = _hermite_coeffs(a)
    x = np.linspace(0.0, 9.0, 12001)
    g = _poly(c, x)
    norm2 = np.sqrt(np.sum(g * g) * (x[1] - x[0]))
    if norm2 < 1e-300:
        return 1e12
    ratio = np.abs(g) * np.exp(0.5 * x * x) / (norm2 * norm2)
    k = 200.0
    m = np.max(ratio)
    if not np.isfinite(m):
        return 1e12
    soft = t + (1.0 / k) * np.log(np.sum(np.exp(k * (ratio - t))))
    return soft


def _optimize_from(a0, k=60.0):
    """Local optimization of (a, t) minimizing t + (1/k) logsumexp(k(ratio-t))."""
    from scipy.optimize import minimize
    x = np.linspace(0.0, 9.0, 10001)
    dx = x[1] - x[0]

    def obj(z):
        a = z[:3]
        t = z[3]
        c = _hermite_coeffs(a)
        g = _poly(c, x)
        norm2 = np.sqrt(np.sum(g * g) * dx)
        if norm2 < 1e-300:
            return 1e12
        ratio = np.abs(g) * np.exp(0.5 * x * x) / (norm2 * norm2)
        m = np.max(ratio)
        if not np.isfinite(m):
            return 1e12
        return t + (1.0 / k) * np.log(np.sum(np.exp(k * (ratio - m)) - k * (t - m)))

    z0 = np.concatenate([a0, [0.0]])
    best_a, best_val = a0.copy(), _bound(_hermite_coeffs(a0))
    for method in ("L-BFGS-B", "trust-constr", "SLSQP"):
        if time.time() > _DEADLINE:
            break
        try:
            res = minimize(obj, z0, method=method,
                           options={"maxiter": 400, "maxfun": 4000,
                                    "ftol": 1e-14, "gtol": 1e-12, "disp": False})
            if res.success or (hasattr(res, "x") and res.x is not None):
                a = res.x[:3]
                v = _bound(_hermite_coeffs(a))
                if np.isfinite(v) and v < best_val:
                    best_val, best_a = v, a.copy()
        except Exception:
            pass
    return best_a, best_val


def construct():
    # Start from the known-good prior state of the art; never regress below it.
    best_a, best_val = _SEED.copy(), _bound(_hermite_coeffs(_SEED))

    # A few deterministic perturbations around the seed for robustness.
    rng = np.random.default_rng(12345)
    starts = [_SEED.copy()]
    for scale in (0.05, 0.02):
        starts.append(_SEED + scale * rng.standard_normal(3))

    for s in starts:
        if time.time() > _DEADLINE - 5.0:
            break
        try:
            a, v = _optimize_from(s)
            if np.isfinite(v) and v < best_val:
                best_val, best_a = v, a.copy()
        except Exception:
            pass

    # Final high-precision polish from the best point.
    if time.time() < _DEADLINE - 5.0:
        try:
            a, v = _optimize_from(best_a, k=120.0)
            if np.isfinite(v) and v < best_val:
                best_val, best_a = v, a.copy()
        except Exception:
            pass

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
