"""Analytic uncertainty bound with precomputed constants + vectorized ops + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # 15s safety margin before 210s kill


def _precompute_constants():
    """Precompute integration constants and derivative matrices once."""
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # (2n)! / (4^n n!) * sqrt(pi)
            log_val = 0.5 * np.log(np.pi)
            for m in range(1, 2*n + 1):
                log_val += np.log(m)
            log_val -= n * np.log(4.0)
            for m in range(1, n + 1):
                log_val -= np.log(m)
            I[n] = np.exp(log_val)
    
    # Build the moment matrix M[k, i, j] = I[k+i+j] for k in {0,1,2}, i,j in 0..6
    # m_k = sum_{i,j} p_i p_j M[k,i,j]
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 15:
                    M[k, i, j] = I[idx]
    
    # For Fourier: n_k = 2*pi * sum_{i,j} q_i q_j M[k,i,j]
    # where q_k = p_k * (-1)^k
    # Precompute sign vector
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Derivative of p w.r.t. (a0, a4, a8)
    # p = D @ a where a = [a0, a4, a8, a12] and a12 = -[1,3,105]/945 @ [a0,a4,a8]
    # So dp/da_c = D[:, c] + D[:, 3] * da12/da_c
    # D maps [a0,a4,a8,a12] to [p0..p6]
    D = np.array([
        [1.0, 3.0, 105.0, 945.0],
        [0.0, -6.0, -420.0, -7560.0],
        [0.0, 1.0, 210.0, 13860.0],
        [0.0, 0.0, -28.0, -6930.0],
        [0.0, 0.0, 1.0, 1155.0],
        [0.0, 0.0, 0.0, -66.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp/da_c for c=0,1,2: shape (7, 3)
    dp_da = np.zeros((7, 3))
    for c in range(3):
        dp_da[:, c] = D[:, c] + D[:, 3] * da12[c]
    
    # dq/da_c: q_k = p_k * signs[k], so dq_k/da_c = dp_k/da_c * signs[k]
    dq_da = dp_da * signs[:, np.newaxis]
    
    return I, M, D, da12, dp_da, dq_da, signs


# Precompute at module level
_I, _M, _D, _da12, _dp_da, _dq_da, _signs = _precompute_constants()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _da12[0]*a0 + _da12[1]*a4 + _da12[2]*a8
    
    # p = D @ [a0, a4, a8, a12]
    a_full = np.array([a0, a4, a8, a12])
    p = _D @ a_full  # shape (7,)
    q = p * _signs  # shape (7,)
    
    # m_k = sum_{i,j} p_i p_j M[k,i,j]
    # Use einsum for speed: m[k] = einsum('i,ijj->k', p, p, M[k])
    # More precisely: m[k] = sum_{i,j} p_i p_j M[k,i,j]
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.einsum('i,ij,j->', p, _M[k], p)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j M[k,i,j]
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.einsum('i,ij,j->', q, _M[k], q)
    
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
    # dm_k/da_c = 2 * sum_i dp_i/da_c * sum_j p_j M[k,i,j]
    # Precompute S[k,i] = sum_j p_j M[k,i,j]
    S_m = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_m[k, i] = np.dot(p, _M[k, i, :])
    
    # dm_k/da_c = 2 * sum_i dp_da[i,c] * S_m[k,i]
    dm = np.zeros((3, 3))  # [k, c]
    for k in range(3):
        for c in range(3):
            dm[k, c] = 2.0 * np.dot(_dp_da[:, c], S_m[k, :])
    
    # Similarly for n: dq_da[i,c]
    S_n = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_n[k, i] = np.dot(q, _M[k, i, :])
    
    dn = np.zeros((3, 3))  # [k, c]
    for k in range(3):
        for c in range(3):
            dn[k, c] = 2.0 * 2.0 * np.pi * np.dot(_dq_da[:, c], S_n[k, :])
    
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/da_c = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    dvm2 = (dm[1, :] * m0 - m2 * dm[0, :]) / (m0 * m0)
    dvn2 = (dn[1, :] * n0 - n2 * dn[0, :]) / (n0 * n0)
    
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
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
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.5, -0.02, 0.0001]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
