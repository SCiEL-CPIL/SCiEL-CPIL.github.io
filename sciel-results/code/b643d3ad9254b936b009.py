"""
Optimized uncertainty bound: fast analytic gradient + multi-start L-BFGS-B
+ Newton/Levenberg-Marquardt high-precision polish.

Design: the objective A(f)*A(F) is a smooth function of 3 free Hermite
coefficients (a0, a4, a8); a12 is fixed by P(0)=0. We compute the bound and
its exact gradient (and, at the end, the exact Hessian) using precomputed
Gaussian moments. A small set of cheap L-BFGS-B runs from diverse starts,
followed by a Newton/LM polish, is far more effective than a slow global
search for this smooth 3D problem.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0


def _hermite_coeffs():
    """
    Polynomial coefficients of H_0, H_4, H_8, H_12.
    Represented as a (4,7) matrix; row i gives coeffs of x^0, x^2, ..., x^12
    for H_{2i}.
    """
    H = np.zeros((4, 7))
    # H_0 = 1
    H[0, 0] = 1.0
    # H_4 = x^4 - 6x^2 + 3
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    # H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    # H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    I[n] = integral x^{2n} e^{-x^2} dx for n=0..12.
    I[n] = (2n-1)!! * sqrt(pi) / 2^n
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2.0 * n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_PI2 = 2.0 * np.pi

# Matrix of moments: M0[i,j] = I[i+j], M2[i,j] = I[1+i+j] (i,j in 0..6)
_idx = np.arange(7)
_M0 = _I[_idx[:, None] + _idx[None, :]]
_M2 = _I[1 + _idx[:, None] + _idx[None, :]]


def _coeffs_from_params(params):
    """Return [a0, a4, a8, a12] with a12 fixed by P(0)=0."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    return np.array([a0, a4, a8, a12])


def _calc_bound_and_grad(params):
    """
    Compute the uncertainty bound A(f)*A(F) and its gradient w.r.t. [a0,a4,a8].
    """
    coeffs = _coeffs_from_params(params)
    p = coeffs @ _H  # shape (7,)
    q = p * _SIGN    # shape (7,)

    # Position moments
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p

    # Frequency moments
    n0 = _PI2 * (q @ _M0 @ q)
    n2 = _PI2 * (q @ _M2 @ q)

    if (m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0)
            or not np.isfinite(m2) or not np.isfinite(n2)):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # dp/da_c = H[c] + (da12/dc) H[3]
    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])
    # dp[c] (7,) for c in 0..2
    dp = _H[0:3, :] + da12[:, None] * _H[3, None, :]  # shape (3,7)
    dq = dp * _SIGN  # shape (3,7)

    # Gradients of moments
    # dm0_c = 2 p^T M0 dp_c ; dm2_c = 2 p^T M2 dp_c
    # dn0_c = 4 pi q^T M0 dq_c ; dn2_c = 4 pi q^T M2 dq_c
    Mp0 = _M0 @ p   # (7,)
    Mp2 = _M2 @ p   # (7,)
    Mq0 = _M0 @ q   # (7,)
    Mq2 = _M2 @ q   # (7,)

    dm0 = 2.0 * (dp @ Mp0)      # (3,)
    dm2 = 2.0 * (dp @ Mp2)      # (3,)
    dn0 = 4.0 * np.pi * (dq @ Mq0)  # (3,)
    dn2 = 4.0 * np.pi * (dq @ Mq2)  # (3,)

    dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)

    grad = (vF * dvm + vf * dvn) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _calc_bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _calc_bound_and_grad(params)
    return val, grad


def _hessian(params):
    """Exact Hessian of the bound w.r.t. [a0,a4,a8]."""
    coeffs = _coeffs_from_params(params)
    p = coeffs @ _H
    q = p * _SIGN

    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    n0 = _PI2 * (q @ _M0 @ q)
    n2 = _PI2 * (q @ _M2 @ q)
    if (m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0)):
        return np.eye(3) * 1e18

    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)

    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])
    dp = _H[0:3, :] + da12[:, None] * _H[3, None, :]  # (3,7)
    dq = dp * _SIGN  # (3,7)

    # First derivatives
    Mp0 = _M0 @ p
    Mp2 = _M2 @ p
    Mq0 = _M0 @ q
    Mq2 = _M2 @ q

    dm0 = 2.0 * (dp @ Mp0)
    dm2 = 2.0 * (dp @ Mp2)
    dn0 = 4.0 * np.pi * (dq @ Mq0)
    dn2 = 4.0 * np.pi * (dq @ Mq2)

    dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)

    # Second derivatives of vf and vF
    # d2vm[c,d] = 2 (dp_c^T M2 dp_d) m0^2 - 2 (dp_c^T M0 dp_d) m2 m0
    #            - 2 m2 (dp_c^T M0 dp_d) m0  ... let's derive properly.
    # vf = m2/m0 ; d vf = (dm2 m0 - m2 dm0)/m0^2
    # d2 vf = 2 (d2m2 m0 - m2 d2m0)/m0^2 - 2 (dm2 m0 - m2 dm0) dm0 / m0^3
    # where d2m2[c,d] = 2 dp_c^T M2 dp_d, d2m0[c,d] = 2 dp_c^T M0 dp_d
    d2m0 = 2.0 * (dp @ _M0.T @ dp.T)   # (3,3)
    d2m2 = 2.0 * (dp @ _M2.T @ dp.T)   # (3,3)
    d2n0 = 4.0 * np.pi * (dq @ _M0.T @ dq.T)  # (3,3)
    d2n2 = 4.0 * np.pi * (dq @ _M2.T @ dq.T)  # (3,3)

    d2vm = (2.0 * (d2m2 * m0 - m2 * d2m0) / (m0 * m0)
            - 2.0 * (dm2[:, None] * m0 - m2 * dm0[:, None]) * dm0[None, :] / (m0 ** 3))
    d2vn = (2.0 * (d2n2 * n0 - n2 * d2n0) / (n0 * n0)
            - 2.0 * (dn2[:, None] * n0 - n2 * dn0[:, None]) * dn0[None, :] / (n0 ** 3))

    # Hessian of bound:
    # b = sqrt(vf vF)
    # db = (vF dvm + vf dvn) / (2 b)
    # d2b = (vF d2vm + vf d2vn + 2 dvm dvn) / (2 b) - (vF dvm + vf dvn)^2 / (4 b^3)
    d2b = ((vF * d2vm + vf * d2vn + 2.0 * dvm[:, None] * dvn[None, :]) / (2.0 * bound)
           - (vF * dvm + vf * dvn)[:, None] * (vF * dvm + vf * dvn)[None, :] / (4.0 * bound ** 3))

    return 0.5 * (d2b + d2b.T)


def _newton_polish(x0, max_iter=300, lam=1e-3):
    """Levenberg-Marquardt / Newton polish using the exact Hessian."""
    x = np.array(x0, dtype=float)
    f, g = _calc_bound_and_grad(x)
    for it in range(max_iter):
        if time.time() > _DEADLINE - 1.0:
            break
        H = _hessian(x)
        # Solve (H + lam I) dx = -g
        try:
            A = H + lam * np.eye(3)
            dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        x_new = x + dx
        f_new, g_new = _calc_bound_and_grad(x_new)
        if not np.isfinite(f_new):
            lam *= 10.0
            continue
        if f_new < f:
            x = x_new
            f, g = f_new, g_new
            lam = max(lam * 0.3, 1e-12)
            if np.linalg.norm(g) < 1e-15:
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                break
    return x, f


def construct():
    """
    Multi-start L-BFGS-B (analytic gradient) from diverse starts, then a
    Newton/Levenberg-Marquardt high-precision polish from the best point.
    """
    # Seed / known-good starting point
    best_params = np.array([-1.13, 0.04, 0.000308642])
    best_val = _objective(best_params)

    # Candidate starts: seed + grid + random
    starts = [best_params.copy()]

    # A small deterministic grid around the seed
    for da0 in (-0.3, 0.0, 0.3):
        for da4 in (-0.02, 0.0, 0.02):
            for da8 in (-0.0002, 0.0, 0.0002):
                starts.append(np.array([-1.13 + da0, 0.04 + da4, 0.000308642 + da8]))

    # A few random starts in a moderate box
    rng = np.random.default_rng(12345)
    for _ in range(20):
        starts.append(np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.02, 0.02),
        ]))

    # Phase 1: L-BFGS-B from each start (cheap, analytic gradient)
    for s in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(
                _objective_grad,
                s,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Newton / Levenberg-Marquardt high-precision polish
    if time.time() < _DEADLINE - 2:
        try:
            x_new, f_new = _newton_polish(best_params)
            if np.isfinite(f_new) and f_new < best_val:
                best_val = f_new
                best_params = x_new.copy()
        except Exception:
            pass

    # Phase 3: final L-BFGS-B from the polished point
    if time.time() < _DEADLINE - 1:
        try:
            res = minimize(
                _objective_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-15},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
