"""Analytic uncertainty bound with global grid search and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _compute_moments(p):
    """
    Given polynomial coefficients p[0..6] for P(x) = sum p_k x^{2k},
    compute moments m_k = integral x^{2k} P(x)^2 e^{-x^2} dx for k=0,1,2
    and Fourier moments n_k = 2*pi * integral x^{2k} Q(x)^2 e^{-x^2} dx
    where Q(x) = P(-ix) = sum p_k (-1)^k x^{2k}.
    """
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pji = p[j]
            if pji == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx <= 12:
                    m[k] += pi * pji * I[idx]
    
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qj = q[j]
            if qj == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx <= 12:
                    n[k] += 2.0 * np.pi * qi * qj * I[idx]
    
    return m, n


def _bound_from_params(params):
    """
    Compute the uncertainty bound from (a0, a4, a8).
    Returns (bound, p) where p is the polynomial coefficient vector.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    m, n = _compute_moments(p)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, p
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, p
    
    return np.sqrt(vf * vF), p


def _objective(params):
    val, _ = _bound_from_params(params)
    return val


def _objective_grad(params):
    """
    Compute objective and gradient using numerical differentiation
    with central differences.
    """
    val, _ = _bound_from_params(params)
    
    h = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        val_plus, _ = _bound_from_params(params_plus)
        val_minus, _ = _bound_from_params(params_minus)
        grad[i] = (val_plus - val_minus) / (2*h)
    
    return val, grad


def construct():
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _objective(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    # Strategy: Global Grid Search on ratios (y=a4/a0, z=a8/a0) with a0=1
    # This avoids the scaling invariance issue and explores the shape space globally.
    
    # 1. Coarse Grid Search
    # Range for y (a4/a0) and z (a8/a0)
    # Seed ratios: y ~ -0.035, z ~ -0.00027
    # We will search a wider range to ensure globality.
    
    y_vals = np.linspace(-0.5, 0.5, 100)
    z_vals = np.linspace(-0.1, 0.1, 100)
    
    best_grid_val = 1e18
    best_y, best_z = 0.0, 0.0
    
    for y in y_vals:
        if time.time() > _DEADLINE - 60:
            break
        for z in z_vals:
            params = np.array([1.0, y, z])
            val, _ = _bound_from_params(params)
            if val < best_grid_val:
                best_grid_val = val
                best_y, best_z = y, z
    
    # Update best if grid found something better
    if best_grid_val < best_val:
        best_val = best_grid_val
        best_params = np.array([1.0, best_y, best_z])
    
    # 2. Refine Grid around the best point
    # Determine local range
    dy = 0.05
    dz = 0.01
    
    y_vals_fine = np.linspace(best_y - dy, best_y + dy, 50)
    z_vals_fine = np.linspace(best_z - dz, best_z + dz, 50)
    
    best_fine_val = 1e18
    best_y_f, best_z_f = best_y, best_z
    
    for y in y_vals_fine:
        if time.time() > _DEADLINE - 30:
            break
        for z in z_vals_fine:
            params = np.array([1.0, y, z])
            val, _ = _bound_from_params(params)
            if val < best_fine_val:
                best_fine_val = val
                best_y_f, best_z_f = y, z
    
    if best_fine_val < best_val:
        best_val = best_fine_val
        best_params = np.array([1.0, best_y_f, best_z_f])
        
    # 3. Local Optimization (L-BFGS-B) from the best grid point
    # We optimize in the original 3D space to allow a0 to vary, though it shouldn't matter much
    # due to homogeneity, it helps the optimizer converge.
    
    start_params = best_params.copy()
    
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, start_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 4. Final Polish with Nelder-Mead
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # 5. One more L-BFGS polish
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
