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

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # Calculate factorial ratio (10-am)! / (10+am)!
        # Using log-gamma for stability
        from scipy.special import gammaln
        log_num = gammaln(_K - am + 1)
        log_den = gammaln(_K + am + 1)
        fact_ratio = np.exp(log_num - log_den)
        
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        
        # lpmv includes Condon-Shortley phase (-1)^m
        # The problem statement says P^m includes Condon-Shortley phase.
        # So we use lpmv directly.
        
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
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _polish(c0, deadline):
    """Polish in real coefficient space using SLSQP with sphere constraint."""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
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
                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            constraint2 = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
            res2 = minimize(fun, best_c, jac=jac, method='SLSQP',
                           constraints=[constraint2],
                           options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    return best_c / nrm, float(best_val)

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 620.0  # 10s safety margin
    
    best_c = None
    best_val = np.inf
    
    # 1. Differential Evolution in real space (broader search)
    bounds = [(-1.0, 1.0) for _ in range(_NC)]
    
    try:
        if time.time() < deadline - 120.0:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
    except Exception:
        pass
    
    # 2. Polish DE result
    if best_c is not None and time.time() < deadline - 60.0:
        polished = _polish(best_c, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 3. Additional restarts from structured starts
    rng = np.random.default_rng(123)
    
    structured_starts = []
    # Pure modes
    for m in range(-_K, _K + 1):
        c0 = np.zeros(_NC)
        c0[m + _K] = 1.0
        structured_starts.append(c0)
    
    # Pairs of modes with same |m|
    for m in range(1, _K + 1):
        idx_pos = m + _K
        idx_neg = -m + _K
        for angle in np.linspace(0, np.pi, 8):
            c0 = np.zeros(_NC)
            c0[idx_pos] = np.cos(angle)
            c0[idx_neg] = np.sin(angle)
            structured_starts.append(c0)
    
    # Random starts
    for _ in range(50):
        c0 = rng.standard_normal(_NC)
        c0 /= np.linalg.norm(c0)
        structured_starts.append(c0)
    
    rng.shuffle(structured_starts)
    
    for c0 in structured_starts:
        if time.time() >= deadline - 30.0:
            break
        polished = _polish(c0, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 4. Final high-precision polish
    if best_c is not None and time.time() < deadline - 10.0:
        final_polished = _polish(best_c, deadline)
        if final_polished:
            c, val = final_polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 5. Fallback
    if best_c is None:
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
    
    best_c = np.asarray(best_c, dtype=np.float64)
    
    # Validation
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
