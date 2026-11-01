"""Uncertainty inequality: tune Hermite-Gaussian test function to minimize A(f)*A(F)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

def _fact_ratio(n):
    """Compute (2n)! / (4^n * n!) * sqrt(pi) = integral x^{2n} e^{-x^2} dx"""
    if n < 0:
        return 0.0
    if n == 0:
        return math.sqrt(math.pi)
    log_val = 0.0
    for m in range(1, 2*n+1):
        log_val += math.log(m)
    log_val -= n * math.log(4.0)
    for m in range(1, n+1):
        log_val -= math.log(m)
    log_val += 0.5 * math.log(math.pi)
    return math.exp(log_val)

# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..12
_I = np.array([_fact_ratio(n) for n in range(13)], dtype=np.float64)

def _bound_and_grad(params):
    """
    Compute uncertainty bound sqrt(vf * vF) and its analytic gradient.
    g(x) = P(x) * exp(-x^2/2) where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    a12 is determined by g(0)=0: a12 = -(a0 + 3*a4 + 105*a8)/945
    
    A(f)^2 = <x^2>_{|g|^2} = m2/m0
    A(F)^2 = <xi^2>_{|F|^2} = n2/n0
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    # |g|^2 = P(x)^2 * e^{-x^2}
    # m_k = integral x^{2k} P(x)^2 e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * _I[idx]
    
    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    # Q(xi) = P(-i*xi) = sum p_k (-1)^k xi^{2k}
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * Q(xi)^2 * e^{-xi^2}
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * math.pi * qq * _I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Analytic gradient
    # da12/d(a0,a4,a8) = (-1/945, -3/945, -105/945)
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c for c=0,1,2 (a0, a4, a8)
    # p[0] = a0 + 3a4 + 105a8 + 945a12
    # p[1] = -6a4 - 420a8 - 7560a12
    # p[2] = a4 + 210a8 + 13860a12
    # p[3] = -28a8 - 6930a12
    # p[4] = a8 + 1155a12
    # p[5] = -66a12
    # p[6] = a12
    
    dp_base = np.array([
        [1.0, 3.0, 105.0],   # dp[0]/d(a0,a4,a8) without a12
        [0.0, -6.0, -420.0],  # dp[1]
        [0.0, 1.0, 210.0],   # dp[2]
        [0.0, 0.0, -28.0],   # dp[3]
        [0.0, 0.0, 1.0],     # dp[4]
        [0.0, 0.0, 0.0],     # dp[5]
        [0.0, 0.0, 0.0],     # dp[6]
    ])
    
    dp_a12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    grad = np.zeros(3)
    for c in range(3):
        # dp_k/da_c = dp_base[k,c] + dp_a12[k] * da12[c]
        dp = dp_base[:, c] + dp_a12 * da12[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j * I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if abs(dpi) < 1e-15:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        # dn_k/da_c = 2*pi * 2 * sum_i dq_i * sum_j q_j * I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if abs(dqi) < 1e-15:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 13:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * 2.0 * math.pi * dqi * s
        
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
    """
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is fixed so that g(0) = 0.
    """
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params

# EVOLVE-BLOCK-END
