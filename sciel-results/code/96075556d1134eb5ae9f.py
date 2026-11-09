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
    dF = 4 * (Y^T(W*f^3) - H*c/||c||^2) / ||c||^4
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _polish_gradient_descent(c, max_iter=500, tol=1e-15):
    """
    High-precision polish using projected gradient descent on the sphere.
    Uses backtracking line search for stability.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()
    
    for it in range(max_iter):
        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        # Project gradient onto tangent space
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break
        
        # Backtracking line search
        step = 1.0
        f_curr = best_val
        for _ in range(30):
            c_trial = c - step * grad_tan
            c_trial = c_trial / np.linalg.norm(c_trial)
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-15 * step * gnorm * gnorm:
                break
            step *= 0.5
        else:
            break
        
        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
        else:
            break
    
    return best_c, best_val


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

    # SLSQP with tighter tolerances
    try:
        res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    # Polish with unconstrained L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            c_uncon = best_c.copy()
            res2 = minimize(_objective_unconstrained, c_uncon, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    # Final high-precision gradient descent polish
    if time.time() < deadline - 2.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=2000, tol=1e-16)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
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

    # Build diverse set of starting points
    rng = np.random.default_rng(42)
    starts = [
        c_init,
        np.array([1 if m % 2 == 0 else 0 for m in range(-_K, _K+1)], dtype=float),
        np.array([1 if m >= 0 else -1 for m in range(-_K, _K+1)], dtype=float),
        np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                  -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=float),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=float),
    ]
    # Add random starts with different seeds
    for seed in [123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909]:
        starts.append(rng.standard_normal(_NC))
    
    # Add some structured random starts
    for seed in [111, 222, 333, 444, 555]:
        c_rand = rng.standard_normal(_NC)
        # Make it more structured by zeroing some components
        mask = rng.random(_NC) > 0.5
        c_rand[mask] = 0.0
        starts.append(c_rand)
    
    # Normalize starts
    for i in range(len(starts)):
        n = np.linalg.norm(starts[i])
        if n > 1e-12:
            starts[i] = starts[i] / n
        else:
            starts[i] = c_init.copy()

    for c0 in starts:
        if time.time() > deadline - 15.0:
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
