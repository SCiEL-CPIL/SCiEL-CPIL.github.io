import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280

_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # Use scipy.special.gammaln for better numerical stability
            from scipy.special import gammaln
            log_num = gammaln(2*n + 1)
            log_den = n * math.log(4) + gammaln(n + 1)
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        dq = dp * _SIGNS
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    return bound, grad

def _compute_bound_2d(params_2d):
    """
    Optimizes with a0 fixed to 1.
    params_2d = [a4, a8]
    """
    a4, a8 = params_2d[0], params_2d[1]
    a0 = 1.0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _compute_grad_2d(params_2d):
    """
    Computes gradient with respect to a4, a8 assuming a0=1.
    """
    a4, a8 = params_2d[0], params_2d[1]
    a0 = 1.0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    
    # Derivatives of a12
    da12_da4 = -12.0 / 665280.0
    da12_da8 = -1680.0 / 665280.0
    
    # Derivatives of p
    dp_da4 = _H_COEFFS[1] + da12_da4 * _H_COEFFS[3]
    dp_da8 = _H_COEFFS[2] + da12_da8 * _H_COEFFS[3]
    
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    grad = np.zeros(2)
    
    # Gradient w.r.t a4
    dm = 2.0 * np.array([dp_da4 @ _M_SPATIAL[k] @ p for k in range(2)])
    dq = dp_da4 * _SIGNS
    dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
    dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
    dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
    grad[0] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    # Gradient w.r.t a8
    dm = 2.0 * np.array([dp_da8 @ _M_SPATIAL[k] @ p for k in range(2)])
    dq = dp_da8 * _SIGNS
    dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
    dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
    dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
    grad[1] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def construct():
    best_val = np.inf
    best_params_3d = None
    
    # We will optimize in 2D space (a4, a8) with a0=1
    # and then map back.
    
    def _try_optimize_2d(start_2d, method='L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-13):
        nonlocal best_val, best_params_3d
        if time.time() > _DEADLINE - 10:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _compute_bound_2d(x), start_2d, method='L-BFGS-B',
                              jac=lambda x: _compute_grad_2d(x)[1], options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Powell':
                res = minimize(lambda x: _compute_bound_2d(x), start_2d, method='Powell',
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-14})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                # Map back to 3D parameters
                a4, a8 = res.x
                a0 = 1.0
                best_params_3d = np.array([a0, a4, a8])
        except Exception:
            pass

    # Phase 1: Multi-start in 2D
    # Based on seed: a0=-1.13, a4=0.04, a8=0.0003
    # Normalized a0=1: a4_norm = 0.04 / -1.13 approx -0.035, a8_norm = 0.0003 / -1.13 approx -0.00027
    # Let's try a range of values.
    
    starts_2d = [
        np.array([-0.035, -0.00027]),
        np.array([0.0, 0.0]),
        np.array([-0.1, -0.001]),
        np.array([0.1, 0.001]),
        np.array([-0.05, -0.0005]),
        np.array([0.05, 0.0005]),
        np.array([-0.2, -0.002]),
        np.array([0.2, 0.002]),
        np.array([-0.01, -0.0001]),
        np.array([0.01, 0.0001]),
        np.array([-0.5, -0.005]),
        np.array([0.5, 0.005]),
        np.array([-0.035, -0.0002]),
        np.array([-0.035, -0.0003]),
        np.array([-0.04, -0.00027]),
        np.array([-0.03, -0.00027]),
    ]
    
    for start in starts_2d:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize_2d(start, 'L-BFGS-B', 5000)
        # Also try Powell for robustness on some starts
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize_2d(start, 'Powell', 2000)

    # Phase 2: Refine best 2D solution
    if best_params_3d is not None and time.time() < _DEADLINE - 20:
        best_2d = best_params_3d[1:]
        # High precision polish
        _try_optimize_2d(best_2d, 'L-BFGS-B', 10000, ftol=1e-16, gtol=1e-14)
        
        # Try small perturbations
        rng = np.random.default_rng(42)
        for scale in [0.1, 0.01, 0.001]:
            if time.time() > _DEADLINE - 10:
                break
            for _ in range(20):
                if time.time() > _DEADLINE - 10:
                    break
                perturbation = scale * np.array([1.0, 0.01]) * rng.normal(0, 1, 2)
                start = best_2d + perturbation
                _try_optimize_2d(start, 'L-BFGS-B', 2000)

    # If we found a solution with a0=1, we can return it.
    # The bound is invariant to sign, so [1, a4, a8] is a valid solution.
    if best_params_3d is not None:
        return best_params_3d
    
    # Fallback to original 3D optimization if 2D failed (unlikely)
    # ... (reuse original logic if needed, but 2D should work)
    return np.array([-1.13, 0.04, 0.000308642])
