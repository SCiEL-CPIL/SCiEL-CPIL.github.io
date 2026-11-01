"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx using recurrence
    # I_0 = sqrt(pi), I_n = (2n-1)/2 * I_{n-1}
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2*n - 1) / 2.0 * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            if pp != 0.0:
                for k in range(3):
                    idx = k + i + j
                    if idx < 14:
                        m[k] += pp * I[idx]
    
    # For Fourier transform: G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    # P(-i*xi) = sum p_k (-1)^k xi^{2k}
    # Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            if qq != 0.0:
                for k in range(3):
                    idx = k + i + j
                    if idx < 14:
                        n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c for each coefficient c=0,1,2
    # p0 = a0+3a4+105a8+945a12
    # p1 = -6a4-420a8-7560a12
    # p2 = a4+210a8+13860a12
    # p3 = -28a8-6930a12
    # p4 = a8+1155a12
    # p5 = -66a12
    # p6 = a12
    
    # dp_k/da_c = coeff[k][c] + da12[c] * dp_k/da12
    # dp_k/da12 = [945, -7560, 13860, -6930, 1155, -66, 1]
    
    dp_da12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    base_coeffs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -6.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = base_coeffs[:, c] + da12[c] * dp_da12
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi != 0.0:
                for k in range(3):
                    s = 0.0
                    for j in range(7):
                        idx = k + i + j
                        if idx < 14:
                            s += p[j] * I[idx]
                    dm[k] += 2.0 * dpi * s
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        # dn_k/da_c = 2*2*pi*sum_i dq_i * sum_j q_j I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi != 0.0:
                for k in range(3):
                    s = 0.0
                    for j in range(7):
                        idx = k + i + j
                        if idx < 14:
                            s += q[j] * I[idx]
                    dn[k] += 2.0 * 2.0*np.pi * dqi * s
        
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
    # Multiple starting points including seed and diverse exploratory points
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
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 2: Nelder-Mead polish
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish with tighter tolerances
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
