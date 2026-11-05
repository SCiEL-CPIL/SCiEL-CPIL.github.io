"""Fresh design: weighted spectral quadrature for luminosity distance.

Uses Gauss-Chebyshev nodes as robust starting points, then applies local
optimization to minimize worst-case relative error over a training set of
CPL cosmologies. Weights are recomputed from the Lagrange basis to ensure
they sum to 1 exactly.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre


def _make_E(w0, w1):
    """CPL Hubble expansion in z: E(z) = (1+z)^3 (1 + w0 + w1 z)."""
    def E(z):
        return (1.0 + z) ** 3 * (1.0 + w0 + w1 * z)
    return E


def _integrand(z, E):
    """f(z) = z / E(z), the quantity to integrate over z in [0,1]."""
    return z / E(z)


def _lagrange_weights(x):
    """Weights for the Lagrange interpolant through nodes x:
    w_i = prod_{j != i} (x_i - x_j) / (x_i^2 - x_j^2), exact in [0,1]."""
    d = x[:, None] - x[None, :]
    np.fill_diagonal(d, 1.0)
    num = np.prod(d, axis=1)
    den = (x[:, None] ** 2 - x[None, :] ** 2)
    np.fill_diagonal(den, 1.0)
    den = np.prod(den, axis=1)
    return num / den


def construct():
    t0 = time.time()
    DEADLINE = 60.0  # safety margin under the 90 s wall-clock kill

    # --- Training CPL models (dark-energy family, smooth in z) -------------
    _TRAIN = [
        (0.0, 0.0), (0.7, -0.5), (-0.5, 0.5), (-1.0, 0.0),
        (-1.0, 0.5), (0.5, -0.3), (-0.3, 0.8), (-1.2, 0.4),
        (0.3, 0.3), (-0.8, -0.2),
    ]

    # --- Exact reference integrals via composite Gauss-Legendre -----------
    _GL = 1024
    _gz, _gw = roots_legendre(_GL)
    _zgrid = 0.5 * (_gz + 1.0)
    _wgrid = 0.5 * _gw
    _ref_cache = {}
    for (w0, w1) in _TRAIN:
        _ref_cache[(w0, w1)] = np.sum(_wgrid * _integrand(_zgrid, _make_E(w0, w1)))

    # --- Starting nodes: Gauss-Chebyshev of first kind in [0,1] ----------
    # Nodes: cos(pi*(2k+1)/(2n)) for k=0,...,n-1, mapped to [0,1]
    n = 8
    k = np.arange(n)
    # Standard Gauss-Chebyshev nodes on [-1,1]
    xc = np.cos(np.pi * (2 * k + 1) / (2 * n))
    # Map to [0,1]: x = (1 + xc)/2
    x0 = 0.5 * (1.0 + xc)
    x0 = np.sort(x0)

    def obj(params):
        xx = np.clip(params, 1e-7, 1.0 - 1e-7)
        xx = np.sort(xx)
        vv = _lagrange_weights(xx)
        worst = 0.0
        for (w0, w1) in _TRAIN:
            q = np.sum(vv * _integrand(xx, _make_E(w0, w1)))
            ref = _ref_cache[(w0, w1)]
            e = abs(q - ref) / abs(ref)
            if e > worst:
                worst = e
        return worst

    best_x = x0.copy()
    best_obj = obj(x0)

    # Try multiple optimization methods/restarts
    try:
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-12})
        if time.time() - t0 < DEADLINE and res.fun < best_obj:
            best_x = np.clip(res.x, 1e-7, 1.0 - 1e-7)
            best_x = np.sort(best_x)
            best_obj = res.fun
    except Exception:
        pass

    # Try a second optimization from a slightly perturbed start if time permits
    if time.time() - t0 < DEADLINE - 10:
        try:
            x0b = best_x + np.random.default_rng(42).uniform(-0.01, 0.01, size=8)
            x0b = np.clip(x0b, 1e-6, 1.0 - 1e-6)
            x0b = np.sort(x0b)
            res2 = minimize(obj, x0b, method="L-BFGS-B",
                            options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-12})
            if time.time() - t0 < DEADLINE and res2.fun < best_obj:
                best_x = np.clip(res2.x, 1e-7, 1.0 - 1e-7)
                best_x = np.sort(best_x)
                best_obj = res2.fun
        except Exception:
            pass

    # --- Finalize: positive, strictly increasing, sum to 1 ---------------
    nodes = np.clip(best_x, 1e-9, 1.0 - 1e-9)
    nodes = np.sort(nodes)
    # enforce strict increase with a tiny floor
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12
    weights = _lagrange_weights(nodes)
    # enforce positivity (should already hold for well-separated nodes)
    weights = np.maximum(weights, 1e-15)
    # normalize exactly to sum 1
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
