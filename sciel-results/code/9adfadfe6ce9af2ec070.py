"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    # Use recurrence: I_n = (2n-1) * I_{n-1} / (4*n) for n>=1, I_0 = sqrt(pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # p[0] = a0 + 3a4 + 105a8 + 945*a12 = 0 by construction
    # p[1] = -6a4 - 420a8 - 7560a12
    # p[2] = a4 + 210a8 + 13860a12
    # p[3] = -28a8 - 6930a12
    # p[4] = a8 + 1155a12
    # p[5] = -66a12
    # p[6] = a12
    p = np.empty(7)
    p[0] = 0.0  # exactly zero by g(0)=0 constraint
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    
    # Vectorized computation of m_k and n_k
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    # Build matrix M where M[i,j] = I[i+j], then m_k = sum_{i,j} p_i p_j I[k+i+j]
    # Equivalent: for each k, m_k = p^T * diag(I[k+0], I[k+1], ..., I[k+12]) * p
    # But simpler: precompute outer product approach
    pp = np.outer(p, p)
    # m[k] = sum_{i,j} pp[i,j] * I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        # Build matrix I_k[i,j] = I[k+i+j] for i,j in 0..6
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    # For Fourier: q_k = p_k * (-1)^k
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # dp/da_c for c=0,1,2 (derivatives of p w.r.t. a0, a4, a8)
    # p[0] = 0 always, so dp[0] = 0
    # p[1] = -6a4 - 420a8 - 7560a12, a12 = -(a0+3a4+105a8)/945
    #   dp1/da0 = -7560 * (-1/945) = 8
    #   dp1/da4 = -6 - 7560*(3/945) = -6 - 24 = -30
    #   dp1/da8 = -420 - 7560*(105/945) = -420 - 840 = -1260
    # p[2] = a4 + 210a8 + 13860a12
    #   dp2/da0 = 13860*(-1/945) = -14.666...
    #   dp2/da4 = 1 + 13860*(3/945) = 1 + 44 = 45
    #   dp2/da8 = 210 + 13860*(105/945) = 210 + 1540 = 1750
    # p[3] = -28a8 - 6930a12
    #   dp3/da0 = -6930*(-1/945) = 7.333...
    #   dp3/da4 = -6930*(3/945) = -22
    #   dp3/da8 = -28 - 6930*(105/945) = -28 - 770 = -798
    # p[4] = a8 + 1155a12
    #   dp4/da0 = 1155*(-1/945) = -1.222...
    #   dp4/da4 = 1155*(3/945) = 3.666...
    #   dp4/da8 = 1 + 1155*(105/945) = 1 + 128.333... = 129.333...
    # p[5] = -66a12
    #   dp5/da0 = -66*(-1/945) = 66/945
    #   dp5/da4 = -66*(3/945) = -198/945
    #   dp5/da8 = -66*(105/945) = -6930/945
    # p[6] = a12
    #   dp6/da0 = -1/945
    #   dp6/da4 = -3/945
    #   dp6/da8 = -105/945
    
    # Precompute S_p and S_q once (they depend on p and q, not on c)
    # S_p[k,i] = sum_j p_j I[k+i+j]
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    # dp/da matrix: shape (3, 7), dp[da_c, i]
    dpda = np.zeros((3, 7))
    dpda[0, 0] = 0.0
    dpda[1, 0] = 0.0
    dpda[2, 0] = 0.0
    # Column 1 (p[1]):
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    # Column 2 (p[2]):
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    # Column 3 (p[3]):
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    # Column 4 (p[4]):
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    # Column 5 (p[5]):
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    # Column 6 (p[6]):
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        
        # dm[k] = 2 * sum_i dp[i] * S_p[k,i]
        dm = 2.0 * (dp @ S_p)
        
        # dq = dp * signs
        dq = dp * signs
        # dn[k] = 4*pi * sum_i dq[i] * S_q[k,i]
        dn = 4.0 * np.pi * (dq @ S_q)
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
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
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-1.7, 0.06, 0.0005]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.4, 0.035, 0.00025]),
        np.array([-1.6, 0.055, 0.00045]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-2.2, 0.07, 0.0006]),
        # Additional diverse starts
        np.array([-1.1, 0.03, 0.00025]),
        np.array([-1.15, 0.05, 0.0004]),
        np.array([-0.9, 0.06, 0.0005]),
        np.array([-1.8, 0.04, 0.0003]),
        np.array([-1.25, 0.07, 0.0006]),
        np.array([-1.35, 0.02, 0.00015]),
        np.array([-0.7, 0.09, 0.0007]),
        np.array([-2.8, 0.1, 0.0008]),
        np.array([-1.55, 0.045, 0.00035]),
        np.array([-1.45, 0.055, 0.00045]),
    ]
    
    rng = np.random.default_rng(42)
    # Add random starts with varying scales
    for scale in [0.3, 0.5, 0.8, 1.2, 1.5]:
        for i in range(4):
            starts.append(rng.normal(0, scale, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: Multi-start L-BFGS-B with moderate tolerance
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish from best
    if best_params is not None and time.time() < _DEADLINE - 40:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 25:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-19, 'fatol': 1e-21, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: One more L-BFGS-B with tightest tolerances if time permits
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
