"""Combined: Fourier-basis representation + direct-space analytic-gradient polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under the 150s wall-clock limit

# Precompute Fourier basis once (DC + K cos/sin modes)
_K = 7
_FREQS = np.arange(1, _K + 1)
_n = np.arange(N)
_ANG = 2.0 * np.pi * _n / N
_BASIS = np.empty((_K + 1, N))
_BASIS[0] = 1.0
for _k in range(_K):
    _BASIS[1 + 2 * _k] = np.cos(_ANG * _FREQS[_k])
    _BASIS[2 + 2 * _k] = np.sin(_ANG * _FREQS[_k])
_NPARAMS = 1 + 2 * _K


def _value(h):
    """Exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _reconstruct(params):
    """h = params @ basis  (DC + cos/sin modes)."""
    return params @ _BASIS


def _smooth_max(conv, beta):
    """Stable log-sum-exp based smooth max."""
    m = np.max(conv)
    return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta


def _objective(params, beta):
    """Scale-invariant surrogate: softmax(conv)/sum(h)^2."""
    h = _reconstruct(params)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e12
    conv = np.convolve(h, h)
    return _smooth_max(conv, beta) / (s * s)


def _objective_grad(params, beta):
    """Analytic gradient of _objective w.r.t params."""
    h = _reconstruct(params)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros_like(params)
    conv = np.convolve(h, h)
    m = np.max(conv)
    expc = np.exp(beta * (conv - m))
    Z = np.sum(expc)
    w = expc / Z                       # softmax weights over conv lags
    # d/dh_i sum_k w_k conv_k = 2 * (w * h)_i  (valid for full conv, h length N)
    g_h = 2.0 * np.convolve(w, h, mode='full')[N - 1:2 * N - 1]
    sm = m + np.log(Z) / beta
    val = sm / (s * s)
    # d val/dh = (g_h/s^2) - (2 sm/s^3)
    g_h = g_h / (s * s) - (2.0 * sm / (s * s * s))
    return _BASIS.T @ g_h


def _polish_direct(h0, beta, maxiter):
    """Refine in full 400-dim h-space using analytic gradient."""
    def f(x):
        s = np.sum(x)
        if abs(s) < 1e-9:
            return 1e12
        conv = np.convolve(x, x)
        return _smooth_max(conv, beta) / (s * s)

    def g(x):
        s = np.sum(x)
        if abs(s) < 1e-9:
            return np.zeros_like(x)
        conv = np.convolve(x, x)
        m = np.max(conv)
        expc = np.exp(beta * (conv - m))
        Z = np.sum(expc)
        w = expc / Z
        g_h = 2.0 * np.convolve(w, x, mode='full')[N - 1:2 * N - 1]
        sm = m + np.log(Z) / beta
        return g_h / (s * s) - (2.0 * sm / (s * s * s))

    try:
        res = minimize(f, h0, jac=g, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if res.x is not None and np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return h0


def construct():
    start = time.time()
    best_h = np.ones(N)
    best_val = _value(best_h)

    def _update(h):
        nonlocal best_h, best_val
        if h is None or not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_h = np.asarray(h, dtype=float).copy()

    rng = np.random.default_rng(12345)

    # Multi-start over the low-dimensional Fourier parameter space
    n_starts = 14
    for i in range(n_starts):
        if time.time() > _DEADLINE - 20.0:
            break
        beta = 25.0
        if i == 0:
            # Start near the all-ones seed: DC = N, rest zero.
            p0 = np.zeros(_NPARAMS)
            p0[0] = N
        else:
            p0 = rng.normal(0.0, 0.25, size=_NPARAMS)
            p0[0] = N * rng.uniform(0.6, 1.4)
        try:
            res = minimize(_objective, p0, jac=_objective_grad,
                           method='L-BFGS-B', args=(beta,),
                           options={'maxiter': 250, 'maxfun': 1250,
                                    'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                _update(_reconstruct(res.x))
        except Exception:
            pass

    # Refinement: direct-space polish on the current best (full resolution)
    if time.time() < _DEADLINE - 5.0:
        _update(_polish_direct(best_h.copy(), beta=40.0, maxiter=300))

    return best_h
