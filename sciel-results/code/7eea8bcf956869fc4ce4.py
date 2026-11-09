# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _objective_L4(c):
    """
    Minimize H(c) = sum(W * f^4) subject to ||c||^2 = 1.
    This is equivalent to minimizing R^4 since ||f||_2^2 = 1.
    """
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    """
    Gradient of H(c) = sum(W * f^4).
    dH/dc = 4 * sum(W * f^3 * Y_i) = 4 * Y^T (W * f^3)
    """
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _objective_unconstrained(c):
    """
    Minimize F(c) = H(c) / ||c||^4, which is scale-invariant.
    This removes the unit sphere constraint.
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return 1e30
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    return H / (norm_sq * norm_sq)


def _gradient_unconstrained(c):
    """
    Gradient of F(c) = H(c) / ||c||^4.
    Using quotient rule:
    dF = (dH * ||c||^4 - H * d(||c||^4)) / ||c||^8
    dH = 4 * Y^T (W * f^3)
    d(||c||^4) = 4 * ||c||^2 * c
    So dF = 4 * (Y^T(W*f^3) - H*c/||c||^2) / ||c||^4
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _optimize_sphere(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    def fun(c):
        return _objective_L4(c)

    def jac(c):
        return _gradient_L4(c)

    def cons_fun(c):
        return np.sum(c * c) - 1.0

    def cons_jac(c):
        return 2.0 * c

    constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]

    best_c = c0.copy()
    best_val = fun(c0)

    # Try SLSQP with tighter tolerances
    try:
        res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    # Polish with unconstrained formulation using L-BFGS-B
    try:
        c_uncon = best_c.copy()
        res2 = minimize(_objective_unconstrained, c_uncon, jac=_gradient_unconstrained,
                        method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = res2.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Initial candidate: All 1s (symmetric)
    c_init = np.ones(_NC)
    c_init /= np.linalg.norm(c_init)
    
    best_c = c_init.copy()
    best_val = _objective_L4(c_init)

    # Try more diverse structured starts + random
    starts = [
        c_init,
        np.array([1 if m % 2 == 0 else 0 for m in range(-_K, _K+1)], dtype=float),
        np.array([1 if m >= 0 else -1 for m in range(-_K, _K+1)], dtype=float),
        np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                  -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=float),
        np.array([0.0]*21, dtype=float),
        np.array([0.0]*21, dtype=float),
    ]
    # Add more random starts
    rng = np.random.default_rng(42)
    for seed in [123, 456, 789, 101, 202, 303, 404, 505]:
        starts.append(rng.standard_normal(_NC))
    
    # Normalize starts
    for i in range(len(starts)):
        n = np.linalg.norm(starts[i])
        if n > 1e-12:
            starts[i] /= n
        else:
            starts[i] = c_init

    for c0 in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            c_opt, val_opt = _optimize_sphere(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = c_init
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    
    return best_c

# EVOLVE-BLOCK-END
