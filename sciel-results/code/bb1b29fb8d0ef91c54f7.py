"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as so

N = 50

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    
    # Fast convolution
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _neg_value_smooth(h, beta=50.0):
    """Negative of a smoothed value for minimization.
    
    Uses a soft-max surrogate for max(g) to provide continuous gradients.
    """
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return 1e10
    h = np.clip(h, 0.0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    
    # Soft-max surrogate for max(g)
    g_max = np.max(g)
    if g_max > 0:
        log_sum_exp = g_max + np.log(np.sum(np.exp(g - g_max)))
        soft_max = log_sum_exp / beta
    else:
        soft_max = 0.0
    
    if norm_1 * soft_max < 1e-30:
        return 1e10
    return -l2 / (norm_1 * soft_max)

def _neg_value_eps(h, eps=1e-6):
    """Negative of value with smoothed max using log-sum-exp with small beta."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return 1e10
    h = np.clip(h, 0.0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    
    # Use log-sum-exp with very small beta for near-exact max
    g_max = np.max(g)
    if g_max > 0:
        # beta = 1/eps gives soft max that approximates max well
        beta = 1.0 / eps
        log_sum_exp = g_max + np.log(np.sum(np.exp(g - g_max)))
        soft_max = log_sum_exp / beta
    else:
        soft_max = 0.0
    
    if norm_1 * soft_max < 1e-30:
        return 1e10
    return -l2 / (norm_1 * soft_max)

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.exp(-2 * x**2)
    h = h / np.max(h)
    return h

def _two_peak_init():
    """Create a two-peak initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.exp(-4 * (x - 0.5)**2) + np.exp(-4 * (x + 0.5)**2)
    h = h / np.max(h)
    return h

def _ramp_init():
    """Create a ramp initial guess."""
    h = np.linspace(0, 1, N)
    return h

def _bump_init(width=0.3):
    """Create a wider bump initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.exp(-1.5 * (x / width)**2)
    h = h / np.max(h)
    return h

def _triangle_init():
    """Create a triangular initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x))
    h = h / np.max(h)
    return h

def _optimize_from_start(h0, deadline):
    """Run L-BFGS-B optimization from a given start."""
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    best_h = np.array(h0, dtype=np.float64)
    best_val = _compute_value(best_h)
    
    # Try multiple beta values, from smooth to sharp
    betas = [5.0, 15.0, 40.0, 100.0, 300.0]
    for beta in betas:
        if time.time() > deadline:
            break
        
        def _neg_smooth(h):
            return _neg_value_smooth(h, beta=beta)
        
        try:
            res = so.minimize(
                _neg_smooth, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            h_opt = res.x
            val_opt = _compute_value(h_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Polish with very smooth surrogate (near-exact max)
    if time.time() < deadline - 2.0:
        try:
            res = so.minimize(
                _neg_value_eps, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 40}
            )
            
            h_polish = res.x
            val_polish = _compute_value(h_polish)
            
            if val_polish > best_val:
                best_val = val_polish
                best_h = h_polish
        except Exception:
            pass
    
    # Final polish with exact objective using tight tolerances
    if time.time() < deadline - 0.5:
        def _neg_exact(h):
            val = _compute_value(h)
            if val < 0:
                return 1e10
            return -val
        
        try:
            res = so.minimize(
                _neg_exact, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 30}
            )
            
            h_polish = res.x
            val_polish = _compute_value(h_polish)
            
            if val_polish > best_val:
                best_val = val_polish
                best_h = h_polish
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    initializations = [
        _hump_init(),
        _two_peak_init(),
        _ramp_init(),
        np.ones(N),
        np.array([0 if i < N//2 else 1 for i in range(N)]),
        _bump_init(0.3),
        _triangle_init(),
    ]
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        h_opt, val_opt = _optimize_from_start(h0, deadline)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    # If we still have time, try more random restarts
    remaining = deadline - time.time()
    if remaining > 10.0:
        rng = np.random.default_rng(42)
        n_restarts = min(6, int(remaining / 15.0))
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            h0 = rng.random(N)
            h0 = h0 / np.max(h0)
            h_opt, val_opt = _optimize_from_start(h0, deadline)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    
    # Final local refinement: small perturbations around best solution
    if time.time() < deadline - 3.0:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.clip(h_pert, 0, None)
            if np.max(h_pert) > 0:
                h_pert = h_pert / np.max(h_pert)
            h_opt, val_opt = _optimize_from_start(h_pert, deadline - 1.0)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
