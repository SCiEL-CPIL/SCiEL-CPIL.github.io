"""Refined Cohn-Goncalves Laguerre construction optimizer for the Fourier sign
uncertainty upper bound (m=12, n=1). Returns 12 positive double-root locations."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import genlaguerre, factorial
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 560.0  # safety margin before the 630 s kill
_M = 12
_N = 1


def _laguerre_matrix(m, x):
    """L_{i,j}(x) for i in [0, m), j in [0, m-1] (degree j)."""
    m = int(m)
    L = np.empty((m, m))
    for j in range(m):
        L[:, j] = genlaguerre(j, m)(x)
    return L


def _build_poly(coeffs, m, n):
    """Return monic-in-Laguerre-basis polynomial P(x) = sum_j c_j L_j^{(m)}(x)
    as a numpy poly1d in standard x-basis (coeffs highest power first)."""
    m = int(m)
    n = int(n)
    deg = m - 1
    # Build P(x) = sum_j c_j L_j^{(m)}(x).
    # Each L_j^{(m)} expressed in x basis.
    poly = np.poly1d([0.0])
    for j in range(deg + 1):
        if coeffs[j] == 0.0:
            continue
        Lj = genlaguerre(j, m)
        poly = poly + coeffs[j] * Lj
    return poly


def _poly_from_roots(roots, m):
    """Given 12 positive double-root locations, build the Laguerre combination
    P(x) that has these as double roots, expressed in Laguerre basis coefficients.

    We solve a linear system: P is a degree-11 Laguerre combination with 12
    double roots -> 24 constraints but only 12 free coefficients, so we instead
    treat the roots as parameters and fit P in Laguerre basis via least squares
    against the double-root conditions (P(r)=0, P'(r)=0)."""
    m = int(m)
    deg = m - 1
    roots = np.asarray(roots, dtype=float)
    # Build Laguerre basis values and derivatives at each root.
    # P(r) = sum_j c_j L_j(r) = 0
    # P'(r) = sum_j c_j L_j'(r) = 0
    A = np.zeros((2 * len(roots), deg + 1))
    for k, r in enumerate(roots):
        Lvals = np.array([genlaguerre(j, m)(r) for j in range(deg + 1)])
        Ldvals = np.array([genlaguerre(j, m).deriv()(r) for j in range(deg + 1)])
        A[2 * k, :] = Lvals
        A[2 * k + 1, :] = Ldvals
    # Least squares: find c minimizing ||A c||  (non-trivial) -> null space
    # We want the null-space vector of A (up to scale). Use SVD.
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    c = Vt[-1, :]
    # Normalize so the leading (highest-degree) Laguerre coefficient is positive
    if c[deg] < 0:
        c = -c
    return c


def _objective_and_grad(roots, m=_M, n=_N):
    """Upper bound C4 and its gradient w.r.t. the 12 root locations.

    The bound is computed from the last sign change of the constructed
    Laguerre combination on the positive axis. We use a smooth surrogate:
    C4 = max over a finite grid of a smoothed penalty plus the last-crossing
    location, regularized to keep roots ordered and positive.
    """
    roots = np.asarray(roots, dtype=float)
    m = int(m)
    deg = m - 1

    # Build P in x-basis from Laguerre-basis coefficients fitted to roots.
    c = _poly_from_roots(roots, m)
    P = _build_poly(c, m, n)
    dP = P.deriv()

    # Grid for evaluating the bound.
    xg = np.linspace(1e-6, 400.0, 2000)
    vals = P(xg)
    dvals = dP(xg)

    # Sign changes of P on positive axis.
    signs = np.sign(vals)
    signs[signs == 0] = 1
    changes = np.where(np.diff(signs) != 0)[0]
    if len(changes) == 0:
        last_cross = 400.0
    else:
        last_cross = xg[changes[-1]]

    # Smooth surrogate for the max magnitude of the derivative ratio that
    # controls the uncertainty constant (log-sum-exp soft-max).
    ratio = np.abs(dvals) / (np.abs(vals) + 1e-8)
    ratio = np.clip(ratio, 0.0, 1e6)
    logsum = np.log(np.sum(np.exp(ratio / 20.0))) * 20.0

    # Regularization: keep roots strictly increasing and positive.
    reg = 1e-3 * np.sum(np.exp(-roots / 1.0))          # push away from 0
    order_pen = 0.0
    for i in range(len(roots) - 1):
        d = roots[i + 1] - roots[i]
        if d < 1e-3:
            order_pen += (1e-3 - d) ** 2

    C4 = float(last_cross + logsum + order_pen)

    # Gradient via analytic differentiation of the smooth parts w.r.t. roots.
    # We differentiate the logsum and regularization terms; last_cross is
    # piecewise-constant, so its gradient is zero almost everywhere.
    grad = np.zeros_like(roots)
    # d logsum / d roots_j  = sum_k w_k * d ratio_k / d roots_j
    w = np.exp(ratio / 20.0)
    w = w / np.sum(w)
    # d ratio_k / d roots_j approximated by chain rule through the fitted
    # coefficients c(roots). Use a small analytic finite difference of c.
    eps = 1e-5
    for j in range(len(roots)):
        r_plus = roots.copy(); r_plus[j] += eps
        r_minus = roots.copy(); r_minus[j] -= eps
        c_plus = _poly_from_roots(r_plus, m)
        c_minus = _poly_from_roots(r_minus, m)
        dc = (c_plus - c_minus) / (2 * eps)
        # d P / d c_j = L_j ; d dP / d c_j = L_j'
        dP_dc = np.array([genlaguerre(i, m)(xg) for i in range(deg + 1)])
        ddP_dc = np.array([genlaguerre(i, m).deriv()(xg) for i in range(deg + 1)])
        dvals_droot = ddP_dc.T @ dc
        vals_droot = dP_dc.T @ dc
        # d ratio_k = d(|dP|/(|P|+eps)) 
        denom = np.abs(vals) + 1e-8
        d_ratio = (dvals_droot * denom - np.abs(dvals) * np.sign(vals) * vals_droot) / (denom ** 2)
        grad[j] += np.dot(w, d_ratio) * 20.0
    # Regularization gradient
    grad += 1e-3 * np.exp(-roots / 1.0) * (1.0 / 1.0)
    for i in range(len(roots) - 1):
        d = roots[i + 1] - roots[i]
        if d < 1e-3:
            g = 2 * (1e-3 - d)
            grad[i] += g
            grad[i + 1] -= g

    return C4, grad


def _evaluate_bound(roots):
    """Final clean evaluation of the upper bound C4 for a set of roots."""
    roots = np.asarray(roots, dtype=float)
    m = int(_M)
    c = _poly_from_roots(roots, m)
    P = _build_poly(c, m, _N)
    xg = np.linspace(1e-6, 400.0, 4000)
    vals = P(xg)
    signs = np.sign(vals); signs[signs == 0] = 1
    changes = np.where(np.diff(signs) != 0)[0]
    last_cross = xg[changes[-1]] if len(changes) else 400.0
    return float(last_cross)


def construct():
    global _START, _DEADLINE
    t0 = time.time()
    deadline = t0 + _DEADLINE

    def remaining():
        return deadline - time.time()

    # Alpha-based initialization: spread roots geometrically in a good basin.
    alpha = 1.0
    x0 = 4.0 * np.arange(1, 13) * alpha  # matches seed scale
    x0 = x0.astype(float)

    best_x = x0.copy()
    best_val = _evaluate_bound(best_x)
    if not np.isfinite(best_val):
        best_val = np.inf

    # L-BFGS-B with analytic gradient.
    try:
        def fun(x):
            if time.time() > deadline - 30:
                return best_val, np.zeros_like(x)
            val, g = _objective_and_grad(x)
            if not np.isfinite(val):
                return 1e12, np.zeros_like(x)
            return val, g

        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'maxfun': 400,
                                'gtol': 1e-8, 'iprint': 0, 'maxls': 20})
        if res.x is not None and np.all(res.x > 0):
            v = _evaluate_bound(res.x)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_x = res.x.copy()
    except Exception:
        pass

    # Final high-precision polish with tighter tolerance if time remains.
    if remaining() > 60:
        try:
            def fun2(x):
                if time.time() > deadline - 20:
                    return best_val, np.zeros_like(x)
                val, g = _objective_and_grad(x)
                if not np.isfinite(val):
                    return 1e12, np.zeros_like(x)
                return val, g
            res2 = minimize(fun2, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'gtol': 1e-10, 'iprint': 0})
            if res2.x is not None and np.all(res2.x > 0):
                v = _evaluate_bound(res2.x)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_x = res2.x.copy()
        except Exception:
            pass

    # Ensure 12 positive, strictly increasing roots.
    best_x = np.sort(best_x)
    best_x = np.maximum(best_x, 1e-6)
    # enforce strict increase
    for i in range(1, len(best_x)):
        if best_x[i] <= best_x[i - 1]:
            best_x[i] = best_x[i - 1] + 1e-4

    return [float(v) for v in best_x]
# EVOLVE-BLOCK-END
