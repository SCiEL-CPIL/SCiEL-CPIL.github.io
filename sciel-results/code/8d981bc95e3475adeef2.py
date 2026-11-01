"""Analytic uncertainty bound + exact gradient, driven by L-BFGS-B with Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 180.0
_H12_0 = 10395.0  # H12(0), monic Hermite


def _I(n):
    """Integral of x^(2n) e^{-x^2} over R = (2n)!/(4^n n!) * sqrt(pi)."""
    if n < 0:
        return 0.0
    v = np.sqrt(np.pi)
    for k in range(1, n + 1):
        v *= (2.0 * k - 1.0) / (4.0 * k)
    return v


_I = np.array([_I(n) for n in range(26)])


def _coeffs(a0, a4, a8):
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / _H12_0
    p = np.array([
        a0 + 3.0 * a4 + 105.0 * a8 + _H12_0 * a12,
        -6.0 * a4 - 420.0 * a8 - 7560.0 * a12,
        a4 + 210.0 * a8 + 13860.0 * a12,
        -28.0 * a8 - 6930.0 * a12,
        a8 + 1155.0 * a12,
        -66.0 * a12,
        a12,
    ])
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    return p, q


def _moments(p):
    m0 = m2 = m4 = 0.0
    for i in range(7):
        pi_ = p[i]
        if pi_ == 0.0:
            continue
        for j in range(7):
            s = pi_ * p[j]
            if s == 0.0:
                continue
            m0 += s * _I[i + j]
            m2 += s * _I[i + j + 1]
            m4 += s * _I[i + j + 2]
    return m0, m2, m4


def _bound_and_grad(x):
    a0, a4, a8 = x[0], x[1], x[2]
    p, q = _coeffs(a0, a4, a8)
    m0, m2, m4 = _moments(p)
    n0, n2, n4 = _moments(q)
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    n4 *= 2.0 * np.pi
    if not (np.isfinite(m0) and np.isfinite(n0)) or m0 <= 1e-30 or n0 <= 1e-30:
        return 1e18, np.zeros(3)
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.zeros(3)
    bound = np.sqrt(vf * vF)

    da12 = np.array([-1.0, -3.0, -105.0]) / _H12_0
    base = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    dpa = np.array([
        base[:, c] + np.array([_H12_0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0]) * da12[c]
        for c in range(3)
    ])  # shape (3,7)

    smp = np.array([np.dot(p, _I[i + 1:j + 1]) for i, j in
                    ((0, 6), (1, 7), (2, 8))])  # smp[k] = sum_j p_j I[k+1+j]
    smq = np.array([np.dot(q, _I[i + 1:j + 1]) for i, j in
                    ((0, 6), (1, 7), (2, 8))])

    grad = np.zeros(3)
    for c in range(3):
        dp = dpa[c]
        dm0 = 2.0 * np.dot(dp, smp[0])
        dm2 = 2.0 * np.dot(dp, smp[1])
        dm4 = 2.0 * np.dot(dp, smp[2])
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        dn0 = 2.0 * np.pi * 2.0 * np.dot(dq, smq[0])
        dn2 = 2.0 * np.pi * 2.0 * np.dot(dq, smq[1])
        dn4 = 2.0 * np.pi * 2.0 * np.dot(dq, smq[2])
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    return bound, grad


def _objective(x):
    return _bound_and_grad(x)[0]


def _objective_grad(x):
    return _bound_and_grad(x)


def construct():
    seed = np.array([-1.13, 0.04, 1.0 / 3240.0])
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.05, 0.0005]),
    ]
    rng = np.random.default_rng(42)
    for _ in range(15):
        starts.append(rng.normal(0, 0.3, 3))

    best_val = np.inf
    best_x = seed.copy()
    
    # Phase 1: L-BFGS-B with exact gradients from multiple starts
    for s in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B', jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish for high precision
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B refinement from the polished point
    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_x, method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-17})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 4: Very tight Nelder-Mead polish
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x
