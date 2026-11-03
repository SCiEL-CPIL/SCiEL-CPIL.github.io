import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _true_score(h):
    """Compute the true objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _smooth_score(h, beta=30.0):
    """
    Smooth surrogate for the objective.
    Uses soft-max for the maximum absolute value.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    ac = ac[:size]
    
    abs_ac = np.abs(ac)
    # Soft max approximation
    # max(x) ~ (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        smax = 0.0
    else:
        shifted = abs_ac - max_val
        smax = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        
    return smax / (s * s)

def _local_search_true(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local search on the true (non-smooth) objective using coordinate descent.
    """
    h = h.copy()
    best_val = _true_score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            
            # Try small perturbations
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size, 
                         2*step_size, -2*step_size]:
                h[i] = old_val + delta
                new_val = _true_score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Evaluate baseline
    base_score = _true_score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Generate initializations
    inits = []
    inits.append(np.ones(N) / N)
    
    rng = np.random.default_rng(123)
    for _ in range(5):
        inits.append(rng.standard_normal(N))
    
    # Chirp
    x = np.arange(N)
    inits.append(np.cos(2 * np.pi * x**2 / N))
    
    # Binary-like
    inits.append(np.sign(rng.standard_normal(N)))
    
    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 15:
            break
            
        # Normalize initial guess
        s_init = np.sum(h_init)
        if abs(s_init) > 1e-9:
            h_init = h_init / s_init
        else:
            h_init = np.ones(N) / N
            
        try:
            # Use L-BFGS-B with numerical gradient
            # We optimize the smooth score
            res = minimize(
                _smooth_score, 
                h_init, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            h_opt = res.x
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-9:
                h_opt = h_opt / s_opt
            else:
                continue
                
            # Evaluate true score
            true_val = _true_score(h_opt)
            
            if true_val < best_score:
                best_score = true_val
                best_h = h_opt.copy()
                
        except Exception:
            pass
        
    # Local search refinement on the best result found
    if time.time() < deadline - 5:
        h_local, val_local = _local_search_true(best_h, max_passes=5, deadline=deadline-2, step_size=0.05)
        if val_local < best_score:
            best_score = val_local
            best_h = h_local.copy()
            
    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
        
    return best_h
