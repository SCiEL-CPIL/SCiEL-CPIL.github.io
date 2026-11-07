import time
import numpy as np
from scipy.optimize import minimize, nnls

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0

def _value(h):
    """
    Computes the objective value for a given height array h.
    Vectorized for speed.
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    return -_value(h)

def _gaussian(center, sigma, amp=1.0):
    """Generate a Gaussian bump on the N-point grid."""
    x = np.arange(N) / N
    s = max(sigma, 1e-4)
    return amp * np.exp(-((x - center)**2) / (2*s**2))

def _params_to_h(params):
    """
    Convert parameters for K Gaussians to a 50-dim height vector.
    params shape: (3*K,) -> [A1, c1, s1, A2, c2, s2, ...]
    """
    K = len(params) // 3
    h = np.zeros(N)
    x = np.arange(N) / N
    for k in range(K):
        A = params[3*k]
        c = params[3*k + 1]
        s = max(params[3*k + 2], 1e-4)
        h += A * np.exp(-((x - c)**2) / (2*s**2))
    return np.maximum(h, 0)

def _neg_value_params(params):
    h = _params_to_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    return -_value(h)

def _coordinate_descent(h, max_iter=500, tol=1e-10, deadline=None):
    """
    Coordinate descent: optimize one element at a time.
    For fixed h_j, the objective as a function of h_i is:
    g = convolve(h, h), so changing h_i affects g[i-1+i] = g[2i-1] and g[i+j] for all j.
    Actually g[k] = sum_j h[j]*h[k-j], so changing h_i affects all g[k] where k-i is in [0,N-1].
    This is complex. Instead, use a simple coordinate update: for each i, find best h_i.
    """
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0)
    best_v = _value(h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try to optimize h[i] by scanning a range
            old_val = h[i]
            # Use golden section or simple scan
            # The objective is not unimodal in h[i] generally, but let's try a local search
            # Actually, for fixed other h_j, g depends quadratically on h[i].
            # Let's just do a small local search around current value.
            best_local = old_val
            best_local_v = best_v
            # Try a few scales
            for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                trial = old_val * scale
                h[i] = trial
                v = _value(h)
                if v > best_local_v:
                    best_local_v = v
                    best_local = trial
            # Also try absolute values if old_val is small
            if old_val < 0.1:
                for trial in [0.1, 0.5, 1.0, 2.0]:
                    h[i] = trial
                    v = _value(h)
                    if v > best_local_v:
                        best_local_v = v
                        best_local = trial
            h[i] = best_local
            if best_local_v > best_v + tol:
                best_v = best_local_v
                improved = True
        if not improved:
            break
    return h, best_v

def _lbfgs_polish(h, deadline=None):
    """
    Full-dimension L-BFGS-B polish.
    """
    bounds_full = [(0, None)] * N
    try:
        res = minimize(
            _neg_value, 
            h, 
            method='L-BFGS-B',
            bounds=bounds_full,
            options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 200
            }
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0, None)
            return h_polished, _value(h_polished)
    except Exception:
        pass
    return h, _value(h)

def _nelder_mead_polish(h, deadline=None):
    """
    Nelder-Mead polish for non-smooth robustness.
    """
    try:
        res = minimize(
            _neg_value,
            h,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-12,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0, None)
            return h_polished, _value(h_polished)
    except Exception:
        pass
    return h, _value(h)

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(42)

    # Initial candidates
    candidates = []
    
    # 1. Uniform seed
    candidates.append(np.ones(N))
    
    # 2. Gaussian bumps with various parameters
    for center in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for sigma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            h = _gaussian(center, sigma, 1.0)
            candidates.append(h)
    
    # 3. Two-Gaussian combinations
    for c1 in [0.2, 0.3, 0.4]:
        for c2 in [0.6, 0.7, 0.8]:
            for s in [0.1, 0.15, 0.2]:
                h = _gaussian(c1, s, 1.0) + _gaussian(c2, s, 1.0)
                candidates.append(h)
    
    # 4. Concentrated shapes
    for beta in [1.2, 1.5, 2.0, 3.0]:
        h = np.array([beta ** i for i in range(N)])
        h = h / h.max()
        candidates.append(h)
    
    # 5. Cosine peaks
    for c in [0.25, 0.5, 0.75]:
        for width in [0.2, 0.3, 0.5]:
            x = np.arange(N) / N
            d = np.minimum(np.abs(x - c), 1.0 - np.abs(x - c))
            d = np.clip(d / width, 0, 1)
            h = np.cos(np.pi * d / 2)**4
            candidates.append(h)
    
    # Evaluate all candidates
    best_h = None
    best_v = 0.0
    for h in candidates:
        h = np.maximum(h, 0)
        v = _value(h)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    # Parametric global search with more Gaussians
    K = 4
    dim = 3 * K
    bounds = [(0, 5), (0, 1), (0.02, 0.5)] * K
    
    n_restarts = 80
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        params0 = np.zeros(dim)
        for k in range(K):
            params0[3*k] = rng.uniform(0.1, 3.0)
            params0[3*k+1] = rng.uniform(0, 1)
            params0[3*k+2] = rng.uniform(0.03, 0.4)
            
        try:
            res = minimize(
                _neg_value_params,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                h_opt = _params_to_h(res.x)
                val_opt = _value(h_opt)
                
                if val_opt > best_v:
                    best_v = val_opt
                    best_h = h_opt.copy()
                    
        except Exception:
            pass
    
    # Also try K=2 and K=3 for diversity
    for K in [2, 3]:
        dim = 3 * K
        bounds = [(0, 5), (0, 1), (0.02, 0.5)] * K
        for i in range(30):
            if time.time() > deadline - 15:
                break
            params0 = np.zeros(dim)
            for k in range(K):
                params0[3*k] = rng.uniform(0.1, 3.0)
                params0[3*k+1] = rng.uniform(0, 1)
                params0[3*k+2] = rng.uniform(0.03, 0.4)
            try:
                res = minimize(
                    _neg_value_params,
                    params0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    h_opt = _params_to_h(res.x)
                    val_opt = _value(h_opt)
                    if val_opt > best_v:
                        best_v = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Coordinate descent on best
    if time.time() < deadline - 10:
        best_h, best_v = _coordinate_descent(best_h, max_iter=300, deadline=deadline - 5)
    
    # L-BFGS-B polish
    if time.time() < deadline - 5:
        best_h, best_v = _lbfgs_polish(best_h, deadline=deadline - 2)
    
    # Nelder-Mead polish for robustness
    if time.time() < deadline - 2:
        best_h_nm, best_v_nm = _nelder_mead_polish(best_h, deadline=deadline - 1)
        if best_v_nm > best_v:
            best_v = best_v_nm
            best_h = best_h_nm
    
    # Final coordinate descent
    if time.time() < deadline - 1:
        best_h, best_v = _coordinate_descent(best_h, max_iter=200, deadline=deadline - 0.5)
    
    # Final safety
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
