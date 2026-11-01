"""Analytic uncertainty bound with vectorized gradient + refined multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0  # 15s safety margin from 210s limit


def _precompute():
    """Precompute all constant arrays needed for bound and gradient computation."""
    # I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2*n - 1) / 2.0 * I[n-1]
    
    # M[k, i, j] = I[k+i+j] for k=0,1,2 and i,j=0..6
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    M[k, i, j] = I[idx]
    
    # sign array for Fourier transform
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # base matrix: base[c, k] = coefficient of a_c in p_k (before a12)
    # p0: a0 + 3a4 + 105a8 + 945a12
    # p1: -6a4 - 420a8 - 7560a12
    # p2: a4 + 210a8 + 13860a12
    # p3: -28a8 - 6930a12
    # p4: a8 + 1155a12
    # p5: -66a12
    # p6: a12
    base = np.zeros((3, 7))
    base[0, 0] = 1.0
    base[1, 0] = 3.0; base[1, 1] = -6.0; base[1, 2] = 1.0
    base[2, 0] = 105.0; base[2, 1] = -420.0; base[2, 2] = 210.0
    base[2, 3] = -28.0; base[2, 4] = 1.0
    
    # da12_coeff[k] = coefficient of a12 in p_k
    da12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # da12 vector: da12/da_c
    da12_vec = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_dc[c, k] = base[c,k] + da12_coeff[k] * da12_vec[c]
    dp_dc = np.zeros((3, 7))
    for c in range(3):
        dp_dc[c, :] = base[c, :] + da12_coeff * da12_vec[c]
    
    # dq_dc[c, k] = dp_dc[c, k] * signs[k]
    dq_dc = dp_dc * signs[np.newaxis, :]
    
    return I, M, signs, base, da12_coeff, da12_vec, dp_dc, dq_dc


# Precompute at module level
_I, _M, _signs, _base, _da12_coeff, _da12_vec, _dp_dc, _dq_dc = _precompute()


def _compute_p(params):
    """Compute polynomial coefficients p[0..6] from (a0, a4, a8)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    p = np.zeros(7)
    for c in range(3):
        p += params[c] * _base[c, :]
    p += a12 * _da12_coeff
    return p


def _bound_and_grad(params):
    """
    Compute uncertainty product bound and its analytic gradient.
    Returns (bound, grad) where bound = sqrt(<x^2>/<1> * <xi^2>/<1>)
    """
    p = _compute_p(params)
    q = p * _signs
    
    # m_k = sum_{i,j} p_i p_j M[k,i,j]  for k=0,1,2
    # Vectorized: m = p^T M_k p for each k
    m = np.zeros(3)
    for k in range(3):
        m[k] = p @ _M[k] @ p
    
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * (q @ _M[k] @ q)
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # dm_k/da_c = 2 * sum_i dp_dc[c,i] * sum_j p_j M[k,i,j]
    #            = 2 * dp_dc[c,:] @ M[k] @ p
    # dn_k/da_c = 2 * 2*pi * dq_dc[c,:] @ M[k] @ q
    
    grad = np.zeros(3)
    for c in range(3):
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * (_dp_dc[c, :] @ _M[k] @ p)
        
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 2.0 * 2.0 * np.pi * (_dq_dc[c, :] @ _M[k] @ q)
        
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
    
    def _time_left():
        return _DEADLINE - time.time()
    
    def try_optimize(start, method='L-BFGS-B', maxiter=3000, ftol=1e-14, gtol=1e-13):
        nonlocal best_val, best_params
        if _time_left() < 20:
            return
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-15,
                                       'adaptive': True})
            else:
                res = minimize(_objective, start, method=method,
                              options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 1: Focused grid search around known good region
    # The seed is around [-1.13, 0.04, 0.000308642]
    # Use a denser grid in the most promising region
    grid_a0 = np.linspace(-1.6, -0.6, 15)
    grid_a4 = np.linspace(0.0, 0.12, 13)
    grid_a8 = np.linspace(-0.001, 0.002, 11)
    
    grid_results = []
    for a0 in grid_a0:
        for a4 in grid_a4:
            for a8 in grid_a8:
                if _time_left() < 50:
                    break
                try:
                    val, _ = _bound_and_grad(np.array([a0, a4, a8]))
                    if np.isfinite(val) and val < 1e6:
                        grid_results.append((val, np.array([a0, a4, a8])))
                except Exception:
                    pass
            if _time_left() < 50:
                break
        if _time_left() < 50:
            break
    
    grid_results.sort(key=lambda x: x[0])
    
    # Stage 2: Multi-start L-BFGS-B
    starts = []
    
    # Always include the seed
    starts.append(np.array([-1.13, 0.04, 0.000308642]))
    
    # Top grid points
    for val, params in grid_results[:25]:
        starts.append(params.copy())
    
    # Structured starts around the seed
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal([-1.13, 0.04, 0.000308642], [0.3, 0.04, 0.0008], 3))
    
    # Broader exploration
    extra_starts = [
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.9, 0.06, 0.0006]),
        np.array([-1.3, 0.035, 0.00025]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
        np.array([-0.8, 0.07, 0.0007]),
        np.array([-1.4, 0.025, 0.00015]),
        np.array([-1.05, 0.055, 0.00045]),
        np.array([-1.25, 0.025, 0.00015]),
    ]
    for s in extra_starts:
        starts.append(s)
    
    # Run L-BFGS-B from each start
    for start in starts:
        if _time_left() < 30:
            break
        try_optimize(start, 'L-BFGS-B', maxiter=2000, ftol=1e-13, gtol=1e-12)
    
    # Stage 3: High-precision polish from best
    if best_params is not None and _time_left() > 20:
        # Nelder-Mead for local refinement (adaptive)
        try_optimize(best_params, 'Nelder-Mead', maxiter=50000, ftol=1e-16)
        
        # Final L-BFGS-B with very tight tolerances
        if _time_left() > 15:
            try_optimize(best_params, 'L-BFGS-B', maxiter=5000, ftol=1e-15, gtol=1e-14)
    
    # Stage 4: Perturbation-based refinement
    if best_params is not None and _time_left() > 10:
        rng2 = np.random.default_rng(777)
        for i in range(8):
            if _time_left() < 8:
                break
            perturbed = best_params + rng2.normal(0, 1e-4, 3)
            try_optimize(perturbed, 'L-BFGS-B', maxiter=1000, ftol=1e-15, gtol=1e-14)
    
    # Stage 5: If time remains, try COBYLA from best as a different local method
    if best_params is not None and _time_left() > 12:
        try_optimize(best_params, 'COBYLA', maxiter=5000, ftol=1e-14)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
