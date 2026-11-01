"""Optimized analytic approach: vectorized bound computation + multi-start L-BFGS-B."""
import numpy as np
import time

_DEADLINE = time.time() + 200  # safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute polynomial coefficients of H_0, H_4, H_8, H_12.
    Represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound(params):
    """
    Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # m_k = int x^{2k} |P|^2 e^{-x^2} dx
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    q = p * _SIGNS
    
    # n_k = 2*pi * int xi^{2k} |Q|^2 e^{-xi^2} dx
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its gradient analytically."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    q = p * _SIGNS
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[1+i+j]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
        
        dq = dp * _SIGNS
        
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += q[j] * _I[i+j]
                s2 += q[j] * _I[1+i+j]
            dn0 += 4.0 * np.pi * dqi * s0
            dn2 += 4.0 * np.pi * dqi * s2
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    return _bound(params)


def _objective_grad(params):
    return _bound_grad(params)


def construct():
    """
    Multi-start optimization with gradient-based polishing.
    Strategy:
    1. Evaluate a set of diverse starting points
    2. For each promising start, run L-BFGS-B with tight tolerances
    3. Return the best result found
    """
    from scipy.optimize import minimize
    
    # Diverse starting points based on known optimal region
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.9, -0.02, -0.001]),
        np.array([-1.1, 0.02, 0.0005]),
        np.array([-1.3, 0.06, 0.002]),
        np.array([-1.05, 0.03, 0.0002]),
        np.array([-1.15, 0.045, 0.0004]),
        np.array([-0.95, -0.01, -0.0005]),
        np.array([-1.25, 0.055, 0.0015]),
    ]
    
    best_val = np.inf
    best_params = starts[0].copy()
    
    # First pass: evaluate all starts and pick the best few for refinement
    start_vals = []
    for s in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            val = _bound(s)
            start_vals.append((val, s))
        except Exception:
            start_vals.append((1e18, s))
    
    start_vals.sort(key=lambda x: x[0])
    
    # Refine the top 5 starting points
    n_refine = min(5, len(start_vals))
    for i in range(n_refine):
        if time.time() > _DEADLINE - 10:
            break
        
        val, s = start_vals[i]
        if val >= best_val:
            continue
        
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={
                              'maxiter': 50000,
                              'ftol': 1e-28,
                              'gtol': 1e-25,
                              'maxls': 500,
                              'iprint': 0
                          })
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If we have time, do a final ultra-tight polish on the best
    if time.time() < _DEADLINE - 5 and best_val < 1e10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={
                              'maxiter': 100000,
                              'ftol': 1e-30,
                              'gtol': 1e-28,
                              'maxls': 1000,
                              'iprint': 0
                          })
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
