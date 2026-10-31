"""Analytic uncertainty bound with exact gradient + multi-start L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(vf * vF) and its analytic gradient.
    
    g(x) = P(x) * exp(-x^2/2) where P(x) = sum_{k=0}^{6} p_k x^{2k}
    The coefficients a0, a4, a8 are free; a12 is fixed so g(0)=0.
    
    A(f)^2 = <x^2>/<1> for |g|^2 = m2/m0
    A(F)^2 = <xi^2>/<1> for |G|^2 = n2/n0
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k for P(x) = sum p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12

    # Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    # Use log-space for numerical stability
    sp = np.sqrt(np.pi)
    I = np.zeros(15)  # need up to n=14 (max index: k+i+j where k<=2, i,j<=6 => max 14)
    for n in range(15):
        if n == 0:
            I[n] = sp
        else:
            # I_n = (2n-1)!! / 2^n * sqrt(pi) = product_{m=1}^{n} (2m-1)/2m * sqrt(pi)
            val = sp
            for m in range(1, n + 1):
                val *= (2.0 * m - 1.0) / (2.0 * m)
            I[n] = val

    # m_k = integral x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi_val = p[i]
        if pi_val == 0.0:
            continue
        for j in range(7):
            pij = pi_val * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    m[k] += pij * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * Q(xi) * e^{-xi^2/2}
    # where Q(xi) = P(-i*xi) = sum p_k (-1)^k xi^{2k}
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)

    # n_k = 2*pi * integral xi^{2k} |Q|^2 e^{-xi^2} dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi_val = q[i]
        if qi_val == 0.0:
            continue
        for j in range(7):
            qij = qi_val * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    n[k] += 2.0 * np.pi * qij * I[idx]

    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 1e-30 or vF <= 1e-30:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Compute gradient
    # da12/da_c for c in {0,1,2} corresponding to {a0, a4, a8}
    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

    # dp_k/da_c:
    # p0 = a0 + 3a4 + 105a8 + 945a12
    # p1 = -6a4 - 420a8 - 7560a12
    # p2 = a4 + 210a8 + 13860a12
    # p3 = -28a8 - 6930a12
    # p4 = a8 + 1155a12
    # p5 = -66a12
    # p6 = a12
    dp_base = np.array([
        [1.0, 3.0, 105.0],   # p0
        [0.0, -6.0, -420.0], # p1
        [0.0, 1.0, 210.0],   # p2
        [0.0, 0.0, -28.0],   # p3
        [0.0, 0.0, 1.0],     # p4
        [0.0, 0.0, 0.0],     # p5
        [0.0, 0.0, 0.0],     # p6
    ])
    dp_a12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

    grad = np.zeros(3)
    for c in range(3):
        dp = dp_base[:, c] + da12[c] * dp_a12

        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s

        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        # dn_k/da_c = 2*pi * 2 * sum_i dq_i * (sum_j q_j I[k+i+j])
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += q[j] * I[idx]
                dn[k] += 4.0 * np.pi * dqi * s

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.045, 0.00045]),
        np.array([-1.8, 0.07, 0.0007]),
        np.array([-2.5, 0.1, 0.001]),
        np.array([0.5, -0.02, 0.0002]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.1, 0.035, 0.00035]),
        np.array([-1.4, 0.055, 0.00055]),
        np.array([-1.6, 0.065, 0.00065]),
        np.array([-1.9, 0.075, 0.00075]),
    ]

    # Add random starts
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))

    best_val = np.inf
    best_params = None

    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)
# EVOLVE-BLOCK-END
