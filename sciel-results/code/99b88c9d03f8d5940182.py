"""Global multi-start optimization of the Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 185.0  # safety margin before 210s kill


def _hermite_coeffs():
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2.0 * n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
# M[i,j] = I[i+j]; N[i,j] = 2*pi*I[i+j]
_M = _I[0:7][:, None] + _I[0:7][None, :]
_M = _I[_M]
_N = 2.0 * np.pi * _M
# N2[i,j] = 2*pi*I[1+i+j]
_M2 = _I[0:7][:, None] + _I[1:8][None, :]
_M2 = _I[_M2]
_N2 = 2.0 * np.pi * _M2
# sign vector for Q
_S = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
# a12 coefficients
_DA12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])


def _bound(params):
    a0, a4, a8 = params
    a12 = _DA12[0] * a0 + _DA12[1] * a4 + _DA12[2] * a8
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    q = p * _S
    m0 = p @ (_M @ p)
    m2 = p @ (_M2 @ p)
    n0 = q @ (_N @ q)
    n2 = q @ (_N2 @ q)
    if m0 <= 1e-30 or n0 <= 1e-30 or not (np.isfinite(m0) and np.isfinite(n0)):
        return 1e18
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18
    return np.sqrt(vf * vF)


def _bound_grad(params):
    a0, a4, a8 = params
    a12 = _DA12[0] * a0 + _DA12[1] * a4 + _DA12[2] * a8
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    q = p * _S
    m0 = p @ (_M @ p)
    m2 = p @ (_M2 @ p)
    n0 = q @ (_N @ q)
    n2 = q @ (_N2 @ q)
    if m0 <= 1e-30 or n0 <= 1e-30 or not (np.isfinite(m0) and np.isfinite(n0)):
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    bound = np.sqrt(vf * vF)

    # precompute matrix-vector products for p and q
    Mp = _M @ p
    M2p = _M2 @ p
    Nq = _N @ q
    N2q = _N2 @ q
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + _DA12[c] * _H[3, :]
        dq = dp * _S
        dm0 = 2.0 * dp @ Mp
        dm2 = 2.0 * dp @ M2p
        dn0 = 2.0 * dq @ Nq
        dn2 = 2.0 * dq @ N2q
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    return bound, grad


def construct():
    """Global multi-start search for the minimal uncertainty bound."""
    from scipy.optimize import minimize

    best_val = _bound(np.array([-1.13, 0.04, 0.000308642]))
    best_params = np.array([-1.13, 0.04, 0.000308642])

    def try_opt(start, maxiter=8000):
        nonlocal best_val, best_params
        try:
            res = minimize(_bound_grad, start, method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # ---- Build a diverse set of starting points ----
    starts = [np.array([-1.13, 0.04, 0.000308642])]  # seed

    # Lattice over a0, a4, a8
    a0_vals = np.linspace(-3.0, 1.5, 10)
    a4_vals = np.linspace(-0.15, 0.25, 8)
    a8_vals = np.linspace(-0.003, 0.005, 6)
    for a0 in a0_vals:
        for a4 in a4_vals:
            for a8 in a8_vals:
                starts.append(np.array([a0, a4, a8]))

    # Random starts (multiple scales)
    rng = np.random.default_rng(12345)
    for scale in [1.0, 2.0, 0.3, 0.1]:
        for _ in range(40):
            starts.append(rng.normal([0.0, 0.04, 0.0005], [scale, 0.15 * scale, 0.003 * scale]))

    # Coordinate-axis extremes
    for v in [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, 0.5, 1.0]:
        starts.append(np.array([v, 0.0, 0.0]))
    for v in [0.0, 0.02, 0.05, 0.1, 0.2, -0.05, -0.1]:
        starts.append(np.array([-1.0, v, 0.0]))
    for v in [0.0, 0.0002, 0.0005, 0.001, 0.002, -0.001]:
        starts.append(np.array([-1.0, 0.05, v]))

    # ---- Phase 1: coarse multi-start optimization ----
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 20:
            break
        try_opt(start, maxiter=4000)

    # ---- Phase 2: local refinement around best (multi-directional) ----
    if time.time() < _DEADLINE - 15:
        center = best_params.copy()
        for scale in [0.5, 0.2, 0.05, 0.01]:
            if time.time() > _DEADLINE - 10:
                break
            # random perturbations around center
            for _ in range(12):
                if time.time() > _DEADLINE - 10:
                    break
                trial = center + rng.normal(0, scale, 3)
                try_opt(trial, maxiter=3000)
            # axis perturbations
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 10:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try_opt(trial, maxiter=3000)

    # ---- Phase 3: high-precision polish ----
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_bound_grad, best_params, method='L-BFGS-B', jac=True,
                           options={'maxiter': 30000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # ---- Phase 4: Nelder-Mead fine polish ----
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_bound, best_params, method='Nelder-Mead',
                           options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # ---- Phase 5: final ultra-precise L-BFGS-B ----
    if time.time() < _DEADLINE - 1:
        try:
            res = minimize(_bound_grad, best_params, method='L-BFGS-B', jac=True,
                           options={'maxiter': 50000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 300})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
