"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute integral table I[n] = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi)
_I_TABLE = np.zeros(28)
_sp = np.sqrt(np.pi)
_I_TABLE[0] = _sp
for _n in range(1, 28):
    _I_TABLE[_n] = _I_TABLE[_n-1] * (2*_n - 1) / (4*_n)

# Precompute the derivative matrix for p_k w.r.t. (a0, a4, a8)
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12
# da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945

_DAP = np.zeros((7, 3))
# Column 0: d/da0
_DAP[0, 0] = 1.0 + 945.0 * (-1.0/945.0)  # = 0
_DAP[1, 0] = 0.0 - 7560.0 * (-1.0/945.0)  # = 8
_DAP[2, 0] = 0.0 + 13860.0 * (-1.0/945.0)  # = -14.666...
_DAP[3, 0] = 0.0 - 6930.0 * (-1.0/945.0)   # = 7.333...
_DAP[4, 0] = 0.0 + 1155.0 * (-1.0/945.0)   # = -1.222...
_DAP[5, 0] = 0.0 - 66.0 * (-1.0/945.0)     # = 0.0698...
_DAP[6, 0] = 0.0 + 1.0 * (-1.0/945.0)      # = -0.001058...
# Column 1: d/da4
_DAP[0, 1] = 3.0 + 945.0 * (-3.0/945.0)    # = 0
_DAP[1, 1] = -6.0 - 7560.0 * (-3.0/945.0)  # = 18
_DAP[2, 1] = 1.0 + 13860.0 * (-3.0/945.0)  # = -42
_DAP[3, 1] = 0.0 - 6930.0 * (-3.0/945.0)   # = 22
_DAP[4, 1] = 0.0 + 1155.0 * (-3.0/945.0)   # = -3.666...
_DAP[5, 1] = 0.0 - 66.0 * (-3.0/945.0)     # = 0.2095...
_DAP[6, 1] = 0.0 + 1.0 * (-3.0/945.0)      # = -0.003174...
# Column 2: d/da8
_DAP[0, 2] = 105.0 + 945.0 * (-105.0/945.0)  # = 0
_DAP[1, 2] = -420.0 - 7560.0 * (-105.0/945.0)  # = 280
_DAP[2, 2] = 210.0 + 13860.0 * (-105.0/945.0)  # = -1050
_DAP[3, 2] = -28.0 - 6930.0 * (-105.0/945.0)    # = 616
_DAP[4, 2] = 1.0 + 1155.0 * (-105.0/945.0)      # = -122
_DAP[5, 2] = 0.0 - 66.0 * (-105.0/945.0)        # = 7.333...
_DAP[6, 2] = 0.0 + 1.0 * (-105.0/945.0)         # = -0.1111...

# Precompute the sign vector for Fourier transform
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _compute_p(params):
    """Compute polynomial coefficients p_k from (a0, a4, a8)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    return p

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    p = _compute_p(params)
    
    # Compute moments m[k] = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for k in range(3):
        # m[k] = p @ (I[k:14] outer product with p) but more efficiently:
        # m[k] = sum_i sum_j p[i]*p[j]*I[k+i+j]
        # Use the fact that I is precomputed
        m[k] = 0.0
        for i in range(7):
            pi = p[i]
            if pi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 28:
                    m[k] += pi * p[j] * _I_TABLE[idx]
    
    # Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n[k] = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        n[k] = 0.0
        for i in range(7):
            qi = q[i]
            if qi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 28:
                    n[k] += qi * q[j] * _I_TABLE[idx]
        n[k] *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DAP[:, c]
        
        # dm[k]/da_c = 2 * sum_i dp[i] * sum_j p[j] * I[k+i+j]
        dm = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0:
                    continue
                for j in range(7):
                    idx = k + i + j
                    if idx < 28:
                        s += p[j] * _I_TABLE[idx]
            dm[k] = 2.0 * dpi * s if False else 2.0 * s  # Wait, need to sum over i too
        
        dm = np.zeros(3)
        for k in range(3):
            total = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0:
                    continue
                for j in range(7):
                    idx = k + i + j
                    if idx < 28:
                        total += dpi * p[j] * _I_TABLE[idx]
            dm[k] = 2.0 * total
        
        dq = dp * _SIGNS
        dn = np.zeros(3)
        for k in range(3):
            total = 0.0
            for i in range(7):
                dqi = dq[i]
                if dqi == 0.0:
                    continue
                for j in range(7):
                    idx = k + i + j
                    if idx < 28:
                        total += dqi * q[j] * _I_TABLE[idx]
            dn[k] = 2.0 * 2.0 * np.pi * total
        
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
    # Multiple starting points - focused on the basin around the seed
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.1, 0.035, 0.00028]),
        np.array([-1.15, 0.045, 0.00034]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-0.9, 0.06, 0.0005]),
        np.array([-1.3, 0.02, 0.0002]),
        np.array([-1.05, 0.042, 0.00032]),
        np.array([-1.18, 0.038, 0.00029]),
        np.array([-1.12, 0.041, 0.00031]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal([-1.13, 0.04, 0.00031], [0.1, 0.01, 1e-5], 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-precise polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
