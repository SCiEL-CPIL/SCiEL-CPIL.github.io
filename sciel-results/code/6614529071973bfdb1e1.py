# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

# Pre-compute quadrature and basis functions
def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # Compute (10-am)! / (10+am)!
        # This is 1 / ((10-am+1) * ... * (10+am))
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        
        # Condon-Shortley phase is included in lpmv
        P = lpmv(am, _K, x)
        
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
            
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    """R^4 = <f^4> / <f^2>^2"""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    """Analytic gradient of R^4"""
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    
    # d/dc ( <f^4> ) = 4 Y^T (W f^3)
    # d/dc ( <f^2>^2 ) = 4 L2 Y^T (W f)
    # dJ/dc = 4/B^2 Y^T (W f^3) - 4A/B^3 Y^T (W f)
    
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _objective_sym(c_sym):
    """Objective for the symmetric subspace (c_{-m} = 0).
    c_sym has length 11: [c_0, c_1, ..., c_10]
    Maps to full c of length 21.
    """
    c_full = np.zeros(_NC)
    c_full[_K] = c_sym[0] # m=0 is at index 10
    for m in range(1, _K + 1):
        c_full[_K + m] = c_sym[m] # m > 0
        # c_full[_K - m] = 0.0 # m < 0 is 0
    return _objective(c_full)

def _gradient_sym(c_sym):
    """Gradient for the symmetric subspace."""
    c_full = np.zeros(_NC)
    c_full[_K] = c_sym[0]
    for m in range(1, _K + 1):
        c_full[_K + m] = c_sym[m]
    
    grad_full = _gradient(c_full)
    
    grad_sym = np.zeros(len(c_sym))
    grad_sym[0] = grad_full[_K]
    for m in range(1, _K + 1):
        grad_sym[m] = grad_full[_K + m]
    return grad_sym

def _polish(c0, deadline):
    """Refine c0 using SLSQP with sphere constraint"""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    
    # Normalize initial guess
    c = c / nrm
    
    def fun(x):
        return _objective(x)
        
    def jac(x):
        return _gradient(x)
        
    best_c = c.copy()
    best_val = fun(c)
    
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 300, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass
        
    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    return best_c / nrm, float(best_val)

def _polish_sym(c0_sym, deadline):
    """Refine c0_sym using SLSQP in the symmetric subspace"""
    c_sym = np.asarray(c0_sym, dtype=np.float64).copy()
    nrm = np.linalg.norm(c_sym)
    if nrm < 1e-14:
        return None
    
    c_sym = c_sym / nrm
    
    def fun(x):
        return _objective_sym(x)
        
    def jac(x):
        return _gradient_sym(x)
        
    best_c_sym = c_sym.copy()
    best_val = fun(c_sym)
    
    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c_sym, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 300, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c_sym = res.x.copy()
            best_c_sym /= np.linalg.norm(best_c_sym)
    except Exception:
        pass
        
    # Map back to full space
    c_full = np.zeros(_NC)
    c_full[_K] = best_c_sym[0]
    for m in range(1, _K + 1):
        c_full[_K + m] = best_c_sym[m]
        
    nrm = np.linalg.norm(c_full)
    if nrm < 1e-14:
        return None
    return c_full / nrm, float(best_val)

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 610.0  # 20s safety margin
    
    best_c = None
    best_val = np.inf
    
    # 1. Search in the symmetric subspace (11D)
    # This is much faster and likely contains the global minimum
    bounds_sym = [(-10.0, 10.0) for _ in range(_K + 1)]
    
    try:
        if time.time() < deadline - 100.0:
            de_result_sym = differential_evolution(
                _objective_sym,
                bounds_sym,
                seed=42,
                maxiter=1000,
                popsize=20,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result_sym.fun < best_val:
                best_val = de_result_sym.fun
                # Map to full space
                c_full = np.zeros(_NC)
                c_full[_K] = de_result_sym.x[0]
                for m in range(1, _K + 1):
                    c_full[_K + m] = de_result_sym.x[m]
                best_c = c_full.copy()
    except Exception as e:
        pass

    # 2. Polish the symmetric result in the full 21D space
    # This allows breaking symmetry if it helps
    if best_c is not None:
        polished = _polish(best_c, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c

    # 3. If time remains, try asymmetric random restarts in full 21D space
    # This covers the case where the global minimum is not symmetric
    rng = np.random.default_rng(123)
    bounds_full = [(-10.0, 10.0) for _ in range(_NC)]
    
    # Run a shorter DE in full space if time permits
    if time.time() < deadline - 50.0:
        try:
            de_result_full = differential_evolution(
                _objective,
                bounds_full,
                seed=43,
                maxiter=200,
                popsize=15,
                tol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result_full.fun < best_val:
                best_val = de_result_full.fun
                best_c = de_result_full.x.copy()
                
                # Polish this full-space result
                polished = _polish(best_c, deadline)
                if polished:
                    c, val = polished
                    if val < best_val:
                        best_val = val
                        best_c = c
        except Exception:
            pass

    # 4. Additional random restarts in symmetric space if time permits
    while time.time() < deadline - 30.0:
        c0_sym = rng.standard_normal(_K + 1)
        nrm = np.linalg.norm(c0_sym)
        if nrm < 1e-14:
            continue
        c0_sym /= nrm
        
        polished_sym = _polish_sym(c0_sym, deadline)
        if polished_sym:
            c, val = polished_sym
            if val < best_val:
                best_val = val
                best_c = c
                
        # Occasionally try a full-space polish from a symmetric start
        if rng.integers(0, 5) == 0:
            c_full_start = np.zeros(_NC)
            c_full_start[_K] = c0_sym[0]
            for m in range(1, _K + 1):
                c_full_start[_K + m] = c0_sym[m]
            polished_full = _polish(c_full_start, deadline)
            if polished_full:
                c, val = polished_full
                if val < best_val:
                    best_val = val
                    best_c = c

    # Final safety
    if best_c is None:
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    best_c = np.asarray(best_c, dtype=np.float64)
    
    # Validation checks
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END
