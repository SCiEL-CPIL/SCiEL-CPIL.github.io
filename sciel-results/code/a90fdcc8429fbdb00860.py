# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0


def _deadline_ok(margin=30.0):
    return (time.time() - _START) < (_DEADLINE - margin)


def _compute_P_sorted(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] using sorted pair-sums with cumulative weights.
    O(n^2 log n) per call instead of O(n^4).
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-12:
        return 0.0
    p = w / w_sum
    
    # Build all pair sums and their weights
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    pair_weights = p[:, None] * p[None, :]  # (n, n)
    
    # Flatten and sort by pair sum
    ps_flat = pair_sums.ravel()
    pw_flat = pair_weights.ravel()
    
    sort_idx = np.argsort(ps_flat, kind='mergesort')
    ps_sorted = ps_flat[sort_idx]
    pw_sorted = pw_flat[sort_idx]
    
    # Cumulative sum of weights
    cw_sorted = np.cumsum(pw_sorted)
    
    # For each (k, l), threshold T = 2*s_l - s_k
    # Q(T) = sum of pw_sorted where ps_sorted < T
    # Use searchsorted to find the index
    T = 2.0 * s[None, :] - s[:, None]  # (k, l), shape (n, n)
    T_flat = T.ravel()
    
    # searchsorted gives index where T_flat would be inserted to maintain order
    # We want count of ps_sorted < T_flat, so use 'left'
    idx = np.searchsorted(ps_sorted, T_flat, side='left')
    # Q values: if idx == 0, Q = 0; else Q = cw_sorted[idx-1]
    Q_vals = np.zeros(len(idx))
    mask = idx > 0
    Q_vals[mask] = cw_sorted[idx[mask] - 1]
    
    Q_matrix = Q_vals.reshape(n, n)
    
    # P = sum_{k,l} p_k * p_l * Q[k,l]
    P = np.sum(p[:, None] * p[None, :] * Q_matrix)
    return float(P)


def _params_to_sw(params, n):
    """
    Convert parameters (log_support, logit_weights) to support and weights.
    params[0:n] are log(support)
    params[n:2n] are logit(weights) -> weights = sigmoid(params)
    """
    log_s = params[:n]
    logit_w = params[n:2*n]
    
    s = np.exp(log_s)
    # sigmoid for weights
    w = 1.0 / (1.0 + np.exp(-logit_w))
    
    # Ensure validity: support >= 0, weights > 0
    # If any support is nan/inf or weights are 0/1 exactly (underflow/overflow), penalize
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        return None, None
    if np.any(~np.isfinite(w)) or np.any(w <= 0) or np.any(w >= 1):
        return None, None
        
    # Normalize weights to sum to 1 (though sigmoid doesn't sum to 1, we treat w_i as unnormalized mass)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return None, None
    w = w / w_sum
    
    return s, w


def _objective_logit(params, n):
    s, w = _params_to_sw(params, n)
    if s is None:
        return 1.0 # Penalty for invalid parameters
    P = _compute_P_sorted(s, w)
    return -P


def _optimize_n_points_logit(n, n_restarts=20, seed=42):
    """
    Optimize for n points using multi-start L-BFGS-B in log-logit space.
    """
    rng = np.random.RandomState(seed)
    best_val = -np.inf
    best_s = None
    best_w = None
    best_params = None
    
    # Bounds for log_support: [log(1e-4), log(1e4)]
    log_s_min = np.log(1e-4)
    log_s_max = np.log(1e4)
    
    # Bounds for logit_weights: [-10, 10] (covers sigmoid range well)
    logit_w_min = -10.0
    logit_w_max = 10.0
    
    bounds = [(log_s_min, log_s_max)] * n + [(logit_w_min, logit_w_max)] * n
    
    for i in range(n_restarts):
        if not _deadline_ok(margin=60.0):
            break
            
        # Random initialization
        log_s_init = rng.uniform(log_s_min, log_s_max, size=n)
        logit_w_init = rng.uniform(logit_w_min, logit_w_max, size=n)
        x0 = np.concatenate([log_s_init, logit_w_init])
        
        try:
            res = minimize(
                _objective_logit,
                x0,
                args=(n,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < 0.9: # Valid improvement
                s, w = _params_to_sw(res.x, n)
                if s is not None:
                    val = _compute_P_sorted(s, w)
                    if val > best_val:
                        best_val = val
                        best_s = s
                        best_w = w
                        best_params = res.x
        except Exception:
            pass
            
    return best_val, best_s, best_w


def _refine_with_scipy(support, weights, n):
    """
    Refine a solution using scipy.optimize.minimize (L-BFGS-B) on the log-logit space.
    """
    def params_to_sw(params):
        log_s = params[:n]
        logit_w = params[n:2*n]
        s = np.exp(log_s)
        w = 1.0 / (1.0 + np.exp(-logit_w))
        w_sum = w.sum()
        if w_sum < 1e-12:
            return None, None
        w = w / w_sum
        return s, w
    
    def objective(params):
        s, w = params_to_sw(params)
        if s is None:
            return 1.0
        return -_compute_P_sorted(s, w)
    
    # Initial params
    log_s0 = np.log(np.clip(support, 1e-12, None))
    # Convert weights to logit space
    w0 = np.clip(weights, 1e-12, 1-1e-12)
    logit_w0 = np.log(w0 / (1 - w0))
    
    x0 = np.concatenate([log_s0, logit_w0])
    
    bounds = [(np.log(1e-4), np.log(1e4))] * n + [(-10.0, 10.0)] * n
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result.fun < 1.0:
            s, w = params_to_sw(result.x)
            if s is not None:
                val = _compute_P_sorted(s, w)
                return val, s, w
        return 0.0, None, None
    except Exception:
        return 0.0, None, None


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    global _START
    _START = time.time()
    
    best_val = -1.0
    best_support = None
    best_weights = None
    
    # Strategy 1: Multi-start L-BFGS-B in log-logit space for small n
    # n=3 is the simplest non-trivial case. n=4 allows more flexibility.
    for n in [3, 4, 5]:
        if not _deadline_ok(margin=120.0):
            break
        
        # Adjust restarts based on n and time
        remaining = _DEADLINE - (time.time() - _START) - 60.0
        if remaining < 30:
            break
            
        n_restarts = max(5, int(remaining / (n * 0.5))) # Heuristic: ~0.5s per restart per dim?
        if n == 3:
            n_restarts = min(n_restarts, 50)
        elif n == 4:
            n_restarts = min(n_restarts, 30)
        else:
            n_restarts = min(n_restarts, 20)
            
        val, s, w = _optimize_n_points_logit(n, n_restarts=n_restarts, seed=42+n)
        
        if val > best_val:
            best_val = val
            best_support = s
            best_weights = w
        
        # Refine the best solution found so far
        if best_support is not None and _deadline_ok(margin=90.0):
            val_ref, s_ref, w_ref = _refine_with_scipy(best_support, best_weights, len(best_support))
            if val_ref > best_val:
                best_val = val_ref
                best_support = s_ref
                best_weights = w_ref

    # Strategy 2: If time remains, try a few random restarts with DE for n=3/4 as a global backup
    # DE might find basins that local optimizers miss from bad starts, but it's slower.
    if _deadline_ok(margin=120.0):
        # Use a small DE search to find a good starting point for refinement, or just to compete
        for n in [3, 4]:
            if not _deadline_ok(margin=60.0):
                break
                
            log_min = np.log(1e-4)
            log_max = np.log(1e4)
            
            def neg_P_de(params, n=n):
                log_s = params[:n]
                logit_w = params[n:2*n]
                s = np.exp(log_s)
                w = 1.0 / (1.0 + np.exp(-logit_w))
                w_sum = w.sum()
                if w_sum < 1e-12:
                    return 1.0
                w = w / w_sum
                if np.any(s < 0) or np.any(~np.isfinite(s)):
                    return 1.0
                return -_compute_P_sorted(s, w)
            
            lb = np.array([log_min] * n + [-10.0] * n)
            ub = np.array([log_max] * n + [10.0] * n)
            bounds = list(zip(lb, ub))
            
            try:
                result = differential_evolution(
                    neg_P_de,
                    bounds,
                    maxiter=30,
                    popsize=15,
                    tol=1e-6,
                    seed=1000 + n,
                    polish=False,
                    callback=lambda xk, convergence=True: not _deadline_ok(margin=30.0),
                    init='latinhypercube'
                )
                
                if result.fun < 1.0:
                    params = result.x
                    s, w = _params_to_sw(params, n)
                    if s is not None:
                        val = _compute_P_sorted(s, w)
                        if val > best_val:
                            best_val = val
                            best_support = s
                            best_weights = w
            except Exception:
                pass

    # Final refinement
    if best_support is not None and _deadline_ok(margin=30.0):
        n = len(best_support)
        val_ref, s_ref, w_ref = _refine_with_scipy(best_support, best_weights, n)
        if val_ref > best_val:
            best_val = val_ref
            best_support = s_ref
            best_weights = w_ref
    
    # Fallback
    if best_support is None:
        # Use a known good 3-point distribution
        support = np.array([0.0, 1.0, 50.0])
        weights = np.array([0.01, 0.89, 0.10])
        return support, weights
    
    support = np.asarray(best_support, dtype=np.float64).ravel()
    weights = np.asarray(best_weights, dtype=np.float64).ravel()
    
    # Final validation and normalization
    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0, 50.0])
        weights = np.array([0.01, 0.89, 0.10])
    
    # Ensure no negative values
    support = np.clip(support, 0.0, None)
    weights = np.clip(weights, 0.0, None)
    
    w_sum = weights.sum()
    if w_sum <= 1e-9:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = weights / w_sum
    
    return support, weights
# EVOLVE-BLOCK-END
