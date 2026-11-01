"""Uncertainty inequality: tune Hermite-Gaussian test function to minimize A(f)*A(F)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

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

# Precompute Toeplitz matrices for k=0,1,2 (7x7 since p has 7 components)
_NP = 7
_T_MATRICES = []
for _k in range(3):
    _T = np.zeros((_NP, _NP))
    for _i in range(_NP):
        for _j in range(_NP):
            _idx = _k + _i + _j
            if _idx < len(_I):
                _T[_i, _j] = _I[_idx]
    _T_MATRICES.append(_T)

def _compute_moments_vec(coeffs, I_table):
    """Vectorized moment computation: m_k = sum_{i,j} c_i c_j I[k+i+j] for k=0,1,2"""
    n = len(coeffs)
    if n == _NP:
        return np.array([coeffs @ _T_MATRICES[k] @ coeffs for k in range(3)])
    m = np.zeros(3)
    for k in range(3):
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                idx = k + i + j
                if idx < len(I_table):
                    T[i, j] = I_table[idx]
        m[k] = coeffs @ T @ coeffs
    return m

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
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    # |g|^2 = P(x)^2 * e^{-x^2}
    m = np.array([p @ _T_MATRICES[k] @ p for k in range(3)])
    
    # Fourier: Q(xi) = P(-i*xi) = sum p_k (-1)^k xi^{2k}
    sign_q = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * sign_q
    
    # |G|^2 = 2*pi * Q(xi)^2 * e^{-xi^2}
    n = np.array([q @ _T_MATRICES[k] @ q for k in range(3)]) * (2.0 * math.pi)
    
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
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    dp_base = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    
    dp_a12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # Precompute Tp and Tq for all k (shared across all c)
    Tp_all = np.array([_T_MATRICES[k] @ p for k in range(3)])  # shape (3, 7)
    Tq_all = np.array([_T_MATRICES[k] @ q for k in range(3)])  # shape (3, 7)
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dp_base[:, c] + dp_a12 * da12[c]
        dq = dp * sign_q
        
        # dm_k/da_c = 2 * sum_i dp_i * (T_k @ p)_i
        dm = 2.0 * (dp @ Tp_all)  # shape (3,)
        
        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * (T_k @ q)_i
        dn = 2.0 * 2.0 * math.pi * (dq @ Tq_all)  # shape (3,)
        
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
        np.array([-1.3, 0.06, 0.0006]),
        np.array([-0.9, 0.07, 0.0007]),
        np.array([-1.4, 0.09, 0.0009]),
        np.array([-1.1, 0.05, 0.0005]),
        np.array([-1.15, 0.035, 0.0003]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.25, 0.045, 0.00045]),
        np.array([-0.95, 0.055, 0.00055]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    for i in range(10):
        starts.append(rng.normal(0, 2.0, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: least_squares formulation - treat bound as parameter to fit
    # Minimize (bound_target - actual_bound)^2 + penalty for deviation
    if best_params is not None and time.time() < _DEADLINE - 30:
        def _lsq_residuals(params_ext):
            """Residuals for least_squares: [bound - target, 0, 0]"""
            a0, a4, a8, target = params_ext
            bound, _ = _bound_and_grad(np.array([a0, a4, a8]))
            return np.array([bound - target, 0.0, 0.0])
        
        def _lsq_jacobian(params_ext):
            """Jacobian for least_squares"""
            a0, a4, a8, target = params_ext
            _, grad = _bound_and_grad(np.array([a0, a4, a8]))
            return np.array([
                [grad[0], 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ])
        
        # Try several target values near current best
        for target in [best_val * 0.999, best_val * 0.9995, best_val * 0.9999, best_val]:
            if time.time() > _DEADLINE - 15:
                break
            try:
                x0 = np.concatenate([best_params, [target]])
                res = minimize(lambda x: _lsq_residuals(x) @ _lsq_residuals(x), x0,
                             method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
                if res.fun < 1e-20 and np.isfinite(res.fun):
                    # Check if the actual bound improved
                    new_params = res.x[:3]
                    new_bound, _ = _bound_and_grad(new_params)
                    if new_bound < best_val and np.isfinite(new_bound):
                        best_val = new_bound
                        best_params = new_params.copy()
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
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
    
    return best_params

# EVOLVE-BLOCK-END
