"""Analytic uncertainty bound with precomputed constants + basin-hopping optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190  # 20s safety margin before 210s kill


def _precompute_constants():
    """Precompute integration constants and derivative matrices once."""
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_val = 0.5 * np.log(np.pi)
            for m in range(1, 2*n + 1):
                log_val += np.log(m)
            log_val -= n * np.log(4.0)
            for m in range(1, n + 1):
                log_val -= np.log(m)
            I[n] = np.exp(log_val)
    
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 15:
                    M[k, i, j] = I[idx]
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
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
    
    dp_da = np.zeros((7, 3))
    for c in range(3):
        dp_da[:, c] = D[:, c] + D[:, 3] * da12[c]
    
    dq_da = dp_da * signs[:, np.newaxis]
    
    return I, M, D, da12, dp_da, dq_da, signs


_I, _M, _D, _da12, _dp_da, _dq_da, _signs = _precompute_constants()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _da12[0]*a0 + _da12[1]*a4 + _da12[2]*a8
    
    a_full = np.array([a0, a4, a8, a12])
    p = _D @ a_full
    q = p * _signs
    
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.einsum('i,ij,j->', p, _M[k], p)
    
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
    
    S_m = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_m[k, i] = np.dot(p, _M[k, i, :])
    
    dm = np.zeros((3, 3))
    for k in range(3):
        for c in range(3):
            dm[k, c] = 2.0 * np.dot(_dp_da[:, c], S_m[k, :])
    
    S_n = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_n[k, i] = np.dot(q, _M[k, i, :])
    
    dn = np.zeros((3, 3))
    for k in range(3):
        for c in range(3):
            dn[k, c] = 2.0 * 2.0 * np.pi * np.dot(_dq_da[:, c], S_n[k, :])
    
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


def _local_minimize(params, maxiter=5000, ftol=1e-16, gtol=1e-15):
    """Run L-BFGS-B from given point."""
    try:
        res = minimize(_objective_grad, params, method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if np.isfinite(res.fun):
            return res.x.copy(), res.fun
    except Exception:
        pass
    return params.copy(), _objective(params)


def construct():
    # Initial diverse set of starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
    ]
    
    # First: find best local minimum from diverse starts
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        params, val = _local_minimize(start, maxiter=3000, ftol=1e-15, gtol=1e-14)
        if val < best_val:
            best_val = val
            best_params = params
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
        best_val = _objective(best_params)
    
    # Basin hopping: perturb + local minimize + accept
    rng = np.random.default_rng(42)
    temp = 0.01  # initial temperature for Metropolis acceptance
    step_scale = 0.3  # initial perturbation scale
    best_val_bh = best_val
    best_params_bh = best_params.copy()
    current_val = best_val
    current_params = best_params.copy()
    
    max_bh_iters = 200
    
    for it in range(max_bh_iters):
        if time.time() > _DEADLINE - 30:
            break
        
        # Perturb current solution
        step = rng.normal(0, step_scale, 3)
        perturbed = current_params + step
        
        # Local minimize from perturbed point
        new_params, new_val = _local_minimize(perturbed, maxiter=2000, ftol=1e-14, gtol=1e-13)
        
        # Metropolis acceptance
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_params = new_params
            current_val = new_val
            if new_val < best_val_bh:
                best_val_bh = new_val
                best_params_bh = new_params.copy()
        
        # Adaptive step scale and temperature
        if it % 10 == 0:
            step_scale *= 0.95
            temp *= 0.9
        
        # Occasionally kick from a fresh random point to escape deep local minima
        if it % 20 == 10:
            fresh = rng.normal(0, 1.0, 3)
            fresh_params, fresh_val = _local_minimize(fresh, maxiter=1500, ftol=1e-13, gtol=1e-12)
            if fresh_val < current_val:
                current_params = fresh_params
                current_val = fresh_val
                if fresh_val < best_val_bh:
                    best_val_bh = fresh_val
                    best_params_bh = fresh_params.copy()
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 15:
        final_params, final_val = _local_minimize(best_params_bh, maxiter=10000, ftol=1e-18, gtol=1e-17)
        if final_val < best_val_bh:
            best_val_bh = final_val
            best_params_bh = final_params
    
    # One more ultra-precise polish if time allows
    if time.time() < _DEADLINE - 5:
        final_params, final_val = _local_minimize(best_params_bh, maxiter=20000, ftol=1e-20, gtol=1e-19)
        if final_val < best_val_bh:
            best_val_bh = final_val
            best_params_bh = final_params
    
    return np.array(best_params_bh)
# EVOLVE-BLOCK-END
