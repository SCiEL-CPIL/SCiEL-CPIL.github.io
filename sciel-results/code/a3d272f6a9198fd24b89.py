"""Optimized uncertainty bound using Differential Evolution and L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound for given coefficients.
    Returns the bound value.
    """
    # a12 is derived from P(0)=0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Polynomial coefficients p[k] for x^(2k)
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    # Calculate moments
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
        
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            pji = p[j]
            qji = q[j]
            if pji == 0 and qji == 0:
                continue
            
            # m0: k=0
            idx0 = i + j
            if idx0 <= 12:
                m0 += pi * pji * I[idx0]
                n0 += 2.0 * np.pi * qi * qji * I[idx0]
                
            # m2: k=1
            idx1 = 1 + i + j
            if idx1 <= 12:
                m2 += pi * pji * I[idx1]
                n2 += 2.0 * np.pi * qi * qji * I[idx1]
                
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)

def _objective_grad(params):
    a0, a4, a8 = params
    val = _objective(params)
    h = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        val_plus = _objective(p_plus)
        val_minus = _objective(p_minus)
        grad[i] = (val_plus - val_minus) / (2*h)
    return val, grad

def construct():
    # Strategy:
    # 1. Differential Evolution on (a4, a8) with a0 fixed to 1.0
    # 2. L-BFGS-B polish on (a0, a4, a8)
    
    # Bounds for DE - widened to explore more
    bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    
    def de_objective(x):
        a4, a8 = x
        return _compute_bound(1.0, a4, a8)
        
    # Multiple DE runs with different seeds and bounds
    candidates = []
    
    # Run 1: Original bounds
    try:
        result_de = differential_evolution(de_objective, bounds, maxiter=200, popsize=20, seed=42, polish=False, tol=1e-12)
        candidates.append((result_de.fun, result_de.x))
    except Exception:
        pass
    
    # Run 2: Different seed
    try:
        result_de2 = differential_evolution(de_objective, bounds, maxiter=200, popsize=20, seed=123, polish=False, tol=1e-12)
        candidates.append((result_de2.fun, result_de2.x))
    except Exception:
        pass
    
    # Run 3: Different bounds (smaller)
    bounds_small = [(-0.5, 0.5), (-0.2, 0.2)]
    try:
        result_de3 = differential_evolution(de_objective, bounds_small, maxiter=300, popsize=25, seed=7, polish=False, tol=1e-12)
        candidates.append((result_de3.fun, result_de3.x))
    except Exception:
        pass
    
    # Run 4: Different bounds (larger)
    bounds_large = [(-5.0, 5.0), (-2.0, 2.0)]
    try:
        result_de4 = differential_evolution(de_objective, bounds_large, maxiter=150, popsize=15, seed=99, polish=False, tol=1e-12)
        candidates.append((result_de4.fun, result_de4.x))
    except Exception:
        pass
    
    if not candidates:
        # Fallback to seed-like values
        best_a4, best_a8 = 0.04, 0.0003
        best_val = _compute_bound(1.0, best_a4, best_a8)
        best_params = np.array([1.0, best_a4, best_a8])
    else:
        # Pick the best candidate
        candidates.sort(key=lambda x: x[0])
        best_val, best_a4_a8 = candidates[0]
        best_a4, best_a8 = best_a4_a8
        best_params = np.array([1.0, best_a4, best_a8])
    
    # L-BFGS-B Polish with multiple starts
    all_starts = [best_params]
    
    # Add other top candidates as additional starting points
    for val, x in candidates[1:3]:
        if val < 100:  # Only add reasonable candidates
            all_starts.append(np.array([1.0, x[0], x[1]]))
    
    # Also add some heuristic starting points
    all_starts.append(np.array([1.0, 0.0, 0.0]))
    all_starts.append(np.array([1.0, -0.1, 0.01]))
    all_starts.append(np.array([1.0, 0.1, -0.01]))
    
    best_overall = best_val
    best_overall_params = best_params
    
    for start in all_starts:
        try:
            if time.time() >= _DEADLINE - 30:
                break
            res = minimize(_objective_grad, start, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_overall and np.all(np.isfinite(res.x)):
                best_overall = res.fun
                best_overall_params = res.x
        except Exception:
            pass
    
    # Final Nelder-Mead polish on the best
    try:
        if time.time() < _DEADLINE - 10:
            res = minimize(_objective, best_overall_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_overall and np.all(np.isfinite(res.x)):
                best_overall = res.fun
                best_overall_params = res.x
    except Exception:
        pass
        
    if not np.all(np.isfinite(best_overall_params)):
        return np.array([1.0, 0.04, 0.0003])
        
    return np.array(best_overall_params)
# EVOLVE-BLOCK-END
