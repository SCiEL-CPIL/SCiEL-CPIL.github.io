"""Uncertainty inequality optimization with analytic gradient and multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # should be ~0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(vf * vF).
    vf = m2/m0 where m_k = integral x^(2k) |P|^2 e^(-x^2) dx
    vF = n2/n0 where n_k = 2*pi * integral x^(2k) |Q|^2 e^(-x^2) dx, Q_k = (-1)^k p_k
    """
    p = _coeffs_from_params(a0, a4, a8)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs

    m = np.zeros(3)
    n = np.zeros(3)

    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            qij = qi * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18

    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)

    if not np.isfinite(bound):
        return 1e18
    return bound


def _compute_bound_and_grad(params):
    """Compute bound and gradient using central finite differences."""
    a0, a4, a8 = params
    bound = _compute_bound(a0, a4, a8)
    if bound >= 1e17:
        return bound, np.zeros(3)

    h = 1e-6
    grad = np.zeros(3)
    for c in range(3):
        pp = params.copy()
        pm = params.copy()
        pp[c] += h
        pm[c] -= h
        fp = _compute_bound(pp[0], pp[1], pp[2])
        fm = _compute_bound(pm[0], pm[1], pm[2])
        grad[c] = (fp - fm) / (2.0 * h)

    return bound, grad


def _objective(params):
    return _compute_bound(params[0], params[1], params[2])


def _objective_grad(params):
    return _compute_bound_and_grad(params)


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with diverse initializations
    starts = []

    for _ in range(10):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.1, 3)
        starts.append(s)

    for _ in range(30):
        s = np.array([rng.uniform(-3.5, 2.5), rng.uniform(-0.6, 0.6), rng.uniform(-0.015, 0.015)])
        starts.append(s)

    for a0 in np.linspace(-3.0, 1.5, 8):
        for a4 in np.linspace(-0.4, 0.4, 6):
            a8 = 0.0008 * rng.normal()
            starts.append(np.array([a0, a4, a8]))

    # Evaluate all starts
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            val = _compute_bound(s[0], s[1], s[2])
            if np.isfinite(val) and val < 1e17:
                scored.append((val, s))
        except Exception:
            pass

    scored.sort(key=lambda x: x[0])

    if scored:
        best_val = scored[0][0]
        best_params = scored[0][1].copy()

    # Phase 2: L-BFGS-B from top starts
    top_n = min(12, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 50:
            break
        s = scored[i][1]
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish
    if time.time() < _DEADLINE - 20:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17,
                                     'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 10:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass

    # Phase 5: Perturbation and re-optimization
    if time.time() < _DEADLINE - 3:
        try:
            for _ in range(6):
                if time.time() > _DEADLINE - 3:
                    break
                perturbed = best_params + rng.normal(0, 5e-5, 3)
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
