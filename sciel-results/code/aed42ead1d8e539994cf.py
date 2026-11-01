"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Combined approach:
- Structural idea from Inspiration 1: exact analytic moments (position/momentum variance)
  computed via polynomial coefficient expansion and precomputed Gaussian integrals,
  yielding a smooth objective with exact analytic gradient.
- Refinement machinery from Current Program: multi-start L-BFGS-B + Nelder-Mead polish,
  explicit deadline guard, and robust fallbacks.

The uncertainty product is sqrt(vf * vF) where
  vf = m2/m0  (variance of |g|^2)
  vF = n2/n0  (variance of |G|^2, G = F(g))
This avoids the nonsmooth L1 norm and gives a smooth, differentiable objective.
"""

import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0  # Safety margin before 210s kill


def _fact_ratio(n):
    """Compute (2n)! / (4^n * n!) * sqrt(pi) = integral x^{2n} e^{-x^2} dx."""
    if n < 0:
        return 0.0
    if n == 0:
        return np.sqrt(np.pi)
    # Use log-space for stability
    log_val = 0.0
    for m in range(1, 2 * n + 1):
        log_val += np.log(m)
    log_val -= n * np.log(4.0)
    for m in range(1, n + 1):
        log_val -= np.log(m)
    log_val += 0.5 * np.log(np.pi)
    return np.exp(log_val)


# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13
_I = np.array([_fact_ratio(n) for n in range(14)])


def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(vf*vF) and its analytic gradient.
    
    g(x) = P(x) * exp(-x^2/2) where P(x) = sum_{k=0}^{6} p_k x^{2k}
    P is built from Hermite coefficients c0, c4, c8 with c12 fixed by g(0)=0.
    
    a12 = -(a0 + 3*a4 + 105*a8) / 945
    
    Position moments: m_k = integral x^{2k} |P|^2 e^{-x^2} dx
    Momentum moments: n_k = 2*pi * integral x^{2k} |Q|^2 e^{-x^2} dx, Q_k = (-1)^k p_k
    
    vf = m2/m0, vF = n2/n0, bound = sqrt(vf*vF)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Build P(x) = sum_{k=0}^{6} p_k x^{2k}
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

    # Compute moments m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pij * _I[idx]

    # Q_k = (-1)^k p_k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    two_pi = 2.0 * np.pi
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += two_pi * qij * _I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    if m2 <= 0 or n2 <= 0:
        # Variance can't be negative; if it is, something is wrong
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 1e-30 or vF <= 1e-30:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2

    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

    # dp_k/da_c for each free coefficient c in {a0, a4, a8}
    # p[0] = a0 + 3a4 + 105a8 + 945a12
    # p[1] = -6a4 - 420a8 - 7560a12
    # p[2] = a4 + 210a8 + 13860a12
    # p[3] = -28a8 - 6930a12
    # p[4] = a8 + 1155a12
    # p[5] = -66a12
    # p[6] = a12

    dp_da = np.zeros((3, 7))
    # For a0 (c=0):
    dp_da[0, 0] = 1.0 + 945.0 * da12[0]
    dp_da[0, 1] = 0.0 - 7560.0 * da12[0]
    dp_da[0, 2] = 0.0 + 13860.0 * da12[0]
    dp_da[0, 3] = 0.0 - 6930.0 * da12[0]
    dp_da[0, 4] = 0.0 + 1155.0 * da12[0]
    dp_da[0, 5] = 0.0 - 66.0 * da12[0]
    dp_da[0, 6] = 0.0 + 1.0 * da12[0]
    # For a4 (c=1):
    dp_da[1, 0] = 3.0 + 945.0 * da12[1]
    dp_da[1, 1] = -6.0 - 7560.0 * da12[1]
    dp_da[1, 2] = 1.0 + 13860.0 * da12[1]
    dp_da[1, 3] = 0.0 - 6930.0 * da12[1]
    dp_da[1, 4] = 0.0 + 1155.0 * da12[1]
    dp_da[1, 5] = 0.0 - 66.0 * da12[1]
    dp_da[1, 6] = 0.0 + 1.0 * da12[1]
    # For a8 (c=2):
    dp_da[2, 0] = 105.0 + 945.0 * da12[2]
    dp_da[2, 1] = -420.0 - 7560.0 * da12[2]
    dp_da[2, 2] = 210.0 + 13860.0 * da12[2]
    dp_da[2, 3] = -28.0 - 6930.0 * da12[2]
    dp_da[2, 4] = 1.0 + 1155.0 * da12[2]
    dp_da[2, 5] = 0.0 - 66.0 * da12[2]
    dp_da[2, 6] = 0.0 + 1.0 * da12[2]

    # Precompute S_pos[k][i] = sum_j p_j I[k+i+j] for efficient gradient
    # dm_k/da_c = 2 * sum_i dp_i * S_pos[k][i]
    S_pos = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += p[j] * _I[idx]
            S_pos[k, i] = s

    # Precompute S_mom[k][i] = sum_j q_j I[k+i+j]
    S_mom = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += q[j] * _I[idx]
            S_mom[k, i] = s

    grad = np.zeros(3)
    for c in range(3):
        dp = dp_da[c]
        # dm_k/da_c = 2 * sum_i dp_i * S_pos[k][i]
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * np.dot(dp, S_pos[k])

        # dq_k/da_c = (-1)^k * dp_k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)

        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * S_mom[k][i]
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 2.0 * two_pi * np.dot(dq, S_mom[k])

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    """Objective function for optimization."""
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    """Objective with gradient for L-BFGS-B."""
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound on A(f)*A(F(f)).
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()

    # Multiple starting points: seed + structured + random
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0006]),
        np.array([-1.2, 0.03, 0.0003]),
        np.array([-1.4, 0.05, 0.0005]),
        np.array([-0.9, 0.07, 0.0007]),
    ]

    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))

    best_val = np.inf
    best_params = None

    for start in starts:
        if time.time() > _DEADLINE - 25.0:
            break
        try:
            # L-BFGS-B with analytic gradient
            res = minimize(
                _objective_grad, start, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Polish with Nelder-Mead from best found
    if best_params is not None and time.time() < _DEADLINE - 15.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(
                _objective_grad, best_params, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Fallback if nothing worked
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])

    return np.array(best_params)


if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    val, grad = _bound_and_grad(coeffs)
    print(f"Upper bound on C4: {val}")
    print(f"Gradient: {grad}")
