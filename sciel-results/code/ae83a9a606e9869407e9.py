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
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
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

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    best_val = np.inf
    best_params = None
    rng = np.random.default_rng(42)
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-15, gtol=1e-13):
        nonlocal best_val, best_params
        if time.time() > _DEADLINE - 10:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-14})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 1: Broad multi-start with diverse initial points
    diverse_starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-0.3, 0.01, 0.0001]),
    ]
    
    for start in diverse_starts:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, 'L-BFGS-B', 2000)
    
    # Phase 2: Random perturbations around best
    if best_params is not None and time.time() < _DEADLINE - 25:
        for scale in [0.1, 0.05, 0.01, 0.005]:
            for _ in range(8):
                if time.time() > _DEADLINE - 20:
                    break
                perturbation = scale * rng.normal(0, 1, 3)
                start = best_params + perturbation
                _try_optimize(start, 'L-BFGS-B', 1500)
    
    # Phase 3: Change of variables - explore in log-space for a0
    if time.time() < _DEADLINE - 15:
        def _log_param_objective(log_params):
            a0 = np.exp(log_params[0])
            a4 = np.tanh(log_params[1]) * 0.2
            a8 = np.tanh(log_params[2]) * 0.005
            params = np.array([a0, a4, a8])
            val, _ = _compute_bound_and_grad(params)
            return val
        
        def _log_param_grad(log_params):
            a0 = np.exp(log_params[0])
            a4 = np.tanh(log_params[1]) * 0.2
            a8 = np.tanh(log_params[2]) * 0.005
            params = np.array([a0, a4, a8])
            val, grad = _compute_bound_and_grad(params)
            # Chain rule
            dlog_params = np.zeros(3)
            dlog_params[0] = grad[0] * a0
            dlog_params[1] = grad[1] * (1 - np.tanh(log_params[1])**2) * 0.2
            dlog_params[2] = grad[2] * (1 - np.tanh(log_params[2])**2) * 0.005
            return val, dlog_params
        
        log_starts = [
            np.array([np.log(1.13), 0.2, 0.06]),
            np.array([np.log(2.0), 0.1, 0.1]),
            np.array([np.log(0.5), 0.3, 0.04]),
        ]
        for start in log_starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(_log_param_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
                a0 = np.exp(res.x[0])
                a4 = np.tanh(res.x[1]) * 0.2
                a8 = np.tanh(res.x[2]) * 0.005
                params = np.array([a0, a4, a8])
                val, _ = _compute_bound_and_grad(params)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = params.copy()
            except Exception:
                pass
    
    # Phase 4: High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 8:
        _try_optimize(best_params, 'L-BFGS-B', 5000, ftol=1e-16, gtol=1e-14)
    
    # Phase 5: Nelder-Mead final polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        _try_optimize(best_params, 'Nelder-Mead', 10000)
    
    # Phase 6: One more ultra-precise L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 2:
        _try_optimize(best_params, 'L-BFGS-B', 10000, ftol=1e-17, gtol=1e-15)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
