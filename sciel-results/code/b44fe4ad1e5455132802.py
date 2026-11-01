"""Analytic uncertainty bound with vectorized gradient + multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx
_I_cache = None

def _get_I():
    global _I_cache
    if _I_cache is None:
        _I_cache = np.zeros(14)
        _I_cache[0] = np.sqrt(np.pi)
        for n in range(1, 14):
            _I_cache[n] = (2*n - 1) / 2.0 * _I_cache[n-1]
    return _I_cache


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient (vectorized).
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    I = _get_I()
    
    # Build matrix M where M[k, i+j] = I[k+i+j] for k=0,1,2 and i,j=0..6
    # m_k = sum_{i,j} p_i p_j I[k+i+j] = p^T M_k p
    # Build I_mat[k, i, j] = I[k+i+j]
    I_mat = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    I_mat[k, i, j] = I[idx]
    
    # m_k = sum_{i,j} p_i p_j I_mat[k,i,j]
    m = np.array([np.einsum('i,ij,j->', p, I_mat[k], p) for k in range(3)])
    
    # For Fourier: q_k = p_k * (-1)^k
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I_mat[k,i,j]
    n = np.array([2.0*np.pi * np.einsum('i,ij,j->', q, I_mat[k], q) for k in range(3)])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation (vectorized)
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp/da_c for c=0,1,2: shape (3, 7)
    base = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0]
    ])
    da12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # dp[c] = base[:, c] + da12_coeff * da12[c]
    dp = base.T + np.outer(da12, da12_coeff)  # shape (3, 7)
    
    grad = np.zeros(3)
    for c in range(3):
        dp_c = dp[c]  # shape (7,)
        
        # dm_k/da_c = 2 * sum_i dp_c[i] * sum_j p[j] I[k+i+j]
        # = 2 * dp_c^T I_mat[k] p  (for each k)
        dm = np.array([2.0 * np.einsum('i,ij,j->', dp_c, I_mat[k], p) for k in range(3)])
        
        dq_c = dp_c * signs
        dn = np.array([4.0*np.pi * np.einsum('i,ij,j->', dq_c, I_mat[k], q) for k in range(3)])
        
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
    best_val = np.inf
    best_params = None
    
    def try_optimize(start, method='L-BFGS-B', maxiter=3000, ftol=1e-14, gtol=1e-13):
        nonlocal best_val, best_params
        if time.time() > _DEADLINE - 25:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            else:
                res = minimize(_objective, start, method=method,
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 1: Coarse grid search
    grid_a0 = np.linspace(-2.5, 0.5, 13)
    grid_a4 = np.linspace(-0.1, 0.2, 13)
    grid_a8 = np.linspace(-0.002, 0.003, 11)
    
    grid_results = []
    for a0 in grid_a0:
        for a4 in grid_a4:
            for a8 in grid_a8:
                if time.time() > _DEADLINE - 60:
                    break
                try:
                    val, _ = _bound_and_grad(np.array([a0, a4, a8]))
                    if np.isfinite(val) and val < 1e6:
                        grid_results.append((val, np.array([a0, a4, a8])))
                except Exception:
                    pass
            if time.time() > _DEADLINE - 60:
                break
        if time.time() > _DEADLINE - 60:
            break
    
    grid_results.sort(key=lambda x: x[0])
    
    # Stage 2: L-BFGS-B from multiple starts
    starts = [np.array([-1.13, 0.04, 0.000308642])]
    for val, params in grid_results[:20]:
        starts.append(params.copy())
    
    extra_starts = [
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-2.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
    ]
    for s in extra_starts:
        starts.append(s)
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal([-1.1, 0.04, 0.0003], [0.5, 0.05, 0.001], 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try_optimize(start, 'L-BFGS-B', maxiter=2000, ftol=1e-13, gtol=1e-12)
    
    # Stage 3: High-precision polish
    if best_params is not None and time.time() < _DEADLINE - 20:
        try_optimize(best_params, 'Nelder-Mead', maxiter=50000, ftol=1e-16)
        if time.time() < _DEADLINE - 10:
            try_optimize(best_params, 'L-BFGS-B', maxiter=5000, ftol=1e-15, gtol=1e-14)
    
    # Stage 4: Perturbation polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        rng2 = np.random.default_rng(777)
        for i in range(5):
            perturbed = best_params + rng2.normal(0, 1e-4, 3)
            try_optimize(perturbed, 'L-BFGS-B', maxiter=1000, ftol=1e-15, gtol=1e-14)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
