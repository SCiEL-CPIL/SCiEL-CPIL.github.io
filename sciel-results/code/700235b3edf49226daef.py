# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights.
    Uses O(n^2 log n) vectorized approach with sorted pairwise sums.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w = w / w.sum()
    
    # Pairwise sums of support
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    flat_s = s_pair.ravel()
    flat_w = w_pair.ravel()
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    # targets[k, l] = 2*s[l] - s[k]
    targets = 2.0 * s[None, :] - s[:, None]
    flat_targets = targets.ravel()
    
    idx = np.searchsorted(sorted_s, flat_targets, side='left')
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    return P

def _compute_P_fast(support, weights):
    """
    Optimized version: sort support first for potential early termination.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w = w / w.sum()
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    s_pair = s_sorted[:, None] + s_sorted[None, :]
    w_pair = w_sorted[:, None] * w_sorted[None, :]
    
    flat_s = s_pair.ravel()
    flat_w = w_pair.ravel()
    
    order2 = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order2]
    sorted_w = flat_w[order2]
    cum_w = np.cumsum(sorted_w)
    
    targets = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    flat_targets = targets.ravel()
    
    idx = np.searchsorted(sorted_s, flat_targets, side='left')
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * P_X2X3)
    return P

def _neg_P_log(x, support):
    """Negative P in log-weight parameterization."""
    x_shifted = x - x.max()
    e = np.exp(x_shifted)
    w = e / e.sum()
    return -_compute_P(support, w)

def _optimize_weights_fixed_support(support, initial_weights, max_iter=150, ftol=1e-12):
    """Optimize weights for a fixed support using log-parameterization."""
    n = len(support)
    w0 = np.abs(np.asarray(initial_weights, dtype=np.float64))
    if w0.sum() <= 1e-12:
        w0 = np.ones(n) / n
    else:
        w0 = w0 / w0.sum()
    
    best_P = _compute_P(support, w0)
    best_w = w0.copy()
    
    x0 = np.log(w0 + 1e-300)
    
    try:
        result = minimize(_neg_P_log, x0, args=(support,), method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
        x_shifted = result.x - result.x.max()
        e = np.exp(x_shifted)
        w_opt = e / e.sum()
        P_opt = _compute_P(support, w_opt)
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt.copy()
    except Exception:
        pass
    
    return best_w, best_P

def _optimize_support_weights(n, initial_support, initial_weights, 
                               s_bounds=None, max_iter_de=50, popsize=15,
                               maxiter_lm=80, time_check=None, deadline=None):
    """
    Jointly optimize support and weights using differential evolution on 
    (log_support, log_weight) parameters.
    """
    if time_check is not None and time_check():
        return initial_support, initial_weights, _compute_P(initial_support, initial_weights)
    
    s0 = np.abs(np.asarray(initial_support, dtype=np.float64))
    w0 = np.abs(np.asarray(initial_weights, dtype=np.float64))
    if w0.sum() <= 1e-12:
        w0 = np.ones(n) / n
    else:
        w0 = w0 / w0.sum()
    
    # Ensure s0 has no zeros for log
    s0_safe = np.maximum(s0, 1e-10)
    w0_safe = np.maximum(w0, 1e-10)
    w0_safe = w0_safe / w0_safe.sum()
    
    # Parameters: [log(s_0), ..., log(s_{n-1}), log(w_0), ..., log(w_{n-1})]
    # But we need s >= 0, so we use s = exp(x) which is always > 0
    # Actually support can be 0, so let's use s = exp(x) - offset? 
    # Simpler: use s_i = exp(x_i) for i>0, and allow s_0 = 0 by fixing it?
    # Let's just use s_i = max(0, x_i) for support, and log(w_i) for weights.
    
    # Actually, let's parameterize as:
    # s_i = exp(x_i) for all i (all positive), then we can shift later
    # w_i = softmax(y_i)
    
    # For DE, we need bounded parameters.
    # Let's use: x in [log(1e-6), log(1e6)] for support (positive)
    # y in [-10, 10] for log-weights (softmax)
    
    x_lo = np.concatenate([np.full(n, np.log(1e-6)), np.full(n, -10.0)])
    x_hi = np.concatenate([np.full(n, np.log(1e6)), np.full(n, 10.0)])
    
    def _objective(params):
        if time_check is not None and time_check():
            return 1e10
        s_params = params[:n]
        w_params = params[n:]
        
        s = np.exp(s_params)  # all positive
        # Normalize weights via softmax
        w_params_shifted = w_params - w_params.max()
        w = np.exp(w_params_shifted)
        w = w / w.sum()
        
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            return 1e10
        if np.any(s < 0) or np.any(w < 0):
            return 1e10
        
        P = _compute_P(s, w)
        return -P
    
    # Initial point
    x0 = np.concatenate([np.log(s0_safe), np.log(w0_safe + 1e-300)])
    x0 = np.clip(x0, x_lo, x_hi)
    
    try:
        de_result = differential_evolution(
            _objective, 
            bounds=list(zip(x_lo, x_hi)),
            x0=x0,
            maxiter=max_iter_de,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if de_result.success or de_result.fun < -_compute_P(initial_support, initial_weights):
            best_params = de_result.x
            s_opt = np.exp(best_params[:n])
            w_params_opt = best_params[n:]
            w_params_shifted = w_params_opt - w_params_opt.max()
            w_opt = np.exp(w_params_shifted)
            w_opt = w_opt / w_opt.sum()
            
            P_opt = _compute_P(s_opt, w_opt)
            return s_opt, w_opt, P_opt
    except Exception:
        pass
    
    # Fallback: optimize weights only
    w_opt, P_opt = _optimize_weights_fixed_support(initial_support, initial_weights, max_iter=maxiter_lm)
    return initial_support, w_opt, P_opt

def _try_geometric_family(r, n, time_check=None, deadline=None):
    """Try geometric support s_i = r^i with optimized weights."""
    if time_check is not None and time_check():
        return None, None, -1.0
    
    s = np.power(r, np.arange(n, dtype=np.float64))
    # Normalize so max is 1 for numerical stability
    if s.max() > 0:
        s = s / s.max()
    
    w = np.ones(n) / n
    P = _compute_P(s, w)
    
    w_opt, P_opt = _optimize_weights_fixed_support(s, w, max_iter=60)
    return s, w_opt, P_opt

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 30 second safety margin
    
    def _time_check():
        return time.time() > deadline
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Phase 1: Quick exploration with small n and simple supports
    for n in [2, 3, 4, 5, 6, 8, 10]:
        if _time_check():
            break
        
        # Uniform on {0, 1, ..., n-1}
        s = np.arange(n, dtype=np.float64)
        w = np.ones(n) / n
        P = _compute_P(s, w)
        _update_best(s, w, P)
        
        # Optimize weights
        if not _time_check():
            w_opt, P_opt = _optimize_weights_fixed_support(s, w, max_iter=40)
            _update_best(s, w_opt, P_opt)
        
        # Try geometric support
        for r in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            if _time_check():
                break
            s_g, w_g, P_g = _try_geometric_family(r, n, _time_check, deadline)
            if s_g is not None:
                _update_best(s_g, w_g, P_g)
    
    # Phase 2: Joint optimization with DE for small n
    for n in [3, 4, 5, 6]:
        if _time_check():
            break
        
        # Start from best found so far, or from a reasonable initial
        if best_support is not None and len(best_support) == n:
            s0 = best_support.copy()
            w0 = best_weights.copy()
        else:
            s0 = np.linspace(0, 1, n)
            w0 = np.ones(n) / n
        
        s_opt, w_opt, P_opt = _optimize_support_weights(
            n, s0, w0, 
            max_iter_de=30, popsize=12, maxiter_lm=50,
            time_check=_time_check, deadline=deadline
        )
        _update_best(s_opt, w_opt, P_opt)
        
        # Also try from a different initial
        s0_alt = np.sort(np.random.uniform(0, 1, n))
        w0_alt = np.random.dirichlet(np.ones(n))
        s_opt2, w_opt2, P_opt2 = _optimize_support_weights(
            n, s0_alt, w0_alt,
            max_iter_de=25, popsize=10, maxiter_lm=40,
            time_check=_time_check, deadline=deadline
        )
        _update_best(s_opt2, w_opt2, P_opt2)
    
    # Phase 3: Try larger n with weight-only optimization on good supports
    for n in [8, 10, 12, 16, 20]:
        if _time_check():
            break
        
        # Use the best support pattern scaled
        if best_support is not None:
            # Interpolate best support to n points
            s_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(best_support)), best_support)
        else:
            s_interp = np.linspace(0, 1, n)
        
        w = np.ones(n) / n
        P = _compute_P(s_interp, w)
        _update_best(s_interp, w, P)
        
        if not _time_check():
            w_opt, P_opt = _optimize_weights_fixed_support(s_interp, w, max_iter=50)
            _update_best(s_interp, w_opt, P_opt)
        
        # Try exponential decay weights
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            if _time_check():
                break
            w_exp = np.power(np.exp(-alpha * np.arange(n) / n), 1.0)
            w_exp = w_exp / w_exp.sum()
            P_exp = _compute_P(s_interp, w_exp)
            _update_best(s_interp, w_exp, P_exp)
            
            if not _time_check():
                w_opt_exp, P_opt_exp = _optimize_weights_fixed_support(s_interp, w_exp, max_iter=40)
                _update_best(s_interp, w_opt_exp, P_opt_exp)
    
    # Phase 4: Fine-tune the best solution with more iterations
    if best_support is not None and not _time_check():
        n_best = len(best_support)
        # More aggressive DE
        s_opt, w_opt, P_opt = _optimize_support_weights(
            n_best, best_support, best_weights,
            max_iter_de=40, popsize=20, maxiter_lm=100,
            time_check=_time_check, deadline=deadline
        )
        _update_best(s_opt, w_opt, P_opt)
        
        # L-BFGS on weights only with more iterations
        w_opt2, P_opt2 = _optimize_weights_fixed_support(best_support, best_weights, max_iter=200)
        _update_best(best_support, w_opt2, P_opt2)
    
    # Phase 5: Try some structured supports with specific properties
    if not _time_check():
        # Support with a large gap: {0, 1, M} with M large
        for M in [5, 10, 20, 50]:
            for n in [3, 4, 5]:
                if _time_check():
                    break
                s = np.array([0.0, 1.0, M] + [0.5] * (n - 3))
                s = np.sort(s)
                w = np.ones(n) / n
                P = _compute_P(s, w)
                _update_best(s, w, P)
                
                if not _time_check():
                    w_opt, P_opt = _optimize_weights_fixed_support(s, w, max_iter=50)
                    _update_best(s, w_opt, P_opt)
    
    # Final: ensure valid output
    if best_support is None or best_weights is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        best_support = np.abs(best_support)
        best_weights = np.abs(best_weights)
        
        # Remove zero-weight points
        mask = best_weights > 1e-15
        if np.sum(mask) < len(best_weights):
            best_support = best_support[mask]
            best_weights = best_weights[mask]
        
        if len(best_support) < 1:
            n = 10
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
    
    # Final validation
    if len(best_support) != len(best_weights):
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    if len(best_support) < 1:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
