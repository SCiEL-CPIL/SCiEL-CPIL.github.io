# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Pair sums and probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Vectorized threshold lookup
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    return float(P)

def _compute_P_fast(s_sorted, w, sorted_sums, cum_probs_cache=None):
    """Fast P computation with precomputed sorted sums. w is normalized."""
    n = len(s_sorted)
    pair_probs = w[:, None] * w[None, :]
    flat_probs = pair_probs.flatten()
    
    # Need to re-sort probs according to sorted_sums order
    # sorted_sums was sorted from pair_sums which was s_sorted[:,None]+s_sorted[None,:]
    # So we need the same ordering
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    flat_sums = pair_sums.flatten()
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w[:, None] * w[None, :] * inner_sums)
    return float(P)

def _neg_P_weights(params, s_sorted, sorted_sums):
    """Negative P for weight optimization with fixed support."""
    log_w = params
    w = np.exp(log_w - np.max(log_w))
    return -_compute_P_fast(s_sorted, w, sorted_sums)

def _neg_P_full(params, n):
    """Negative P for joint optimization."""
    log_w = params[:n]
    support = params[n:]
    w = np.exp(log_w - np.max(log_w))
    support = np.maximum(support, 0.0)
    return -_compute_P(support, w)

def _optimize_weights_fixed(support, weights, timeout=120, max_iter=800):
    """Optimize weights with fixed support."""
    s_sorted, _ = _sort_support(support)
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    sorted_sums = np.sort(pair_sums.flatten())
    
    init_log_w = np.log(weights + 1e-15)
    best_x = init_log_w.copy()
    best_val = _neg_P_weights(best_x, s_sorted, sorted_sums)
    
    start_time = time.time()
    
    # Powell
    try:
        res = minimize(_neg_P_weights, best_x, args=(s_sorted, sorted_sums),
                     method='Powell', options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time < timeout:
        try:
            res = minimize(_neg_P_weights, best_x, args=(s_sorted, sorted_sums),
                         method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if time.time() - start_time < timeout:
        try:
            bounds = [(-40, 40)] * len(s_sorted)
            res = minimize(_neg_P_weights, best_x, args=(s_sorted, sorted_sums),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    w = np.exp(best_x - np.max(best_x))
    return s_sorted, w, -best_val

def _sort_support(support):
    s = np.asarray(support, dtype=np.float64)
    order = np.argsort(s)
    return s[order], order

def _optimize_joint(support, weights, timeout=120, max_iter=200):
    """Jointly optimize support and weights."""
    n = len(support)
    s_sorted, _ = _sort_support(support)
    init_log_w = np.log(weights + 1e-15)
    init_params = np.concatenate([init_log_w, s_sorted])
    
    best_params = init_params.copy()
    best_val = _neg_P_full(best_params, n)
    
    start_time = time.time()
    
    try:
        res = minimize(_neg_P_full, best_params, args=(n,),
                     method='Powell', options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() - start_time < timeout:
        try:
            res = minimize(_neg_P_full, best_params, args=(n,),
                         method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    log_w = best_params[:n]
    support_out = np.maximum(best_params[n:], 0.0)
    w = np.exp(log_w - np.max(log_w))
    return support_out, w, -best_val

def _local_refine(support, weights, timeout=60):
    """Iterative local refinement: alternate weight optimization and small support perturbations."""
    best_s = support.copy()
    best_w = weights.copy()
    best_P = _compute_P(best_s, best_w)
    start_time = time.time()
    
    for iteration in range(20):
        if time.time() - start_time > timeout:
            break
        
        # Optimize weights
        try:
            s_sorted, w_opt, P = _optimize_weights_fixed(best_s, best_w, timeout=timeout/3, max_iter=300)
            if P > best_P:
                best_P = P
                best_s = s_sorted
                best_w = w_opt
        except Exception:
            pass
        
        if time.time() - start_time > timeout:
            break
        
        # Small support perturbation + weight re-optimization
        n = len(best_s)
        for _ in range(3):
            if time.time() - start_time > timeout:
                break
            perturbation = np.random.randn(n) * 0.05 * (best_s.max() - best_s.min() + 1)
            pert_s = np.maximum(best_s + perturbation, 0.0)
            pert_s = np.sort(pert_s)
            try:
                s_sorted, w_opt, P = _optimize_weights_fixed(pert_s, best_w, timeout=timeout/6, max_iter=200)
                if P > best_P:
                    best_P = P
                    best_s = s_sorted
                    best_w = w_opt
            except Exception:
                pass
    
    return best_s, best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w):
        nonlocal best_support, best_weights, best_P
        P = _compute_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
        return P
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Uniform small
    n1 = 8
    support1 = np.arange(n1, dtype=np.float64)
    weights1 = np.ones(n1) / n1
    initial_configs.append((support1, weights1))
    
    # Config 2: Skewed toward low values
    n2 = 8
    support2 = np.arange(n2, dtype=np.float64)
    weights2 = np.array([0.35, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02])
    weights2 /= weights2.sum()
    initial_configs.append((support2, weights2))
    
    # Config 3: Bimodal
    n3 = 6
    support3 = np.array([0, 1, 2, 4, 6, 8])
    weights3 = np.array([0.3, 0.15, 0.1, 0.2, 0.15, 0.1])
    weights3 /= weights3.sum()
    initial_configs.append((support3, weights3))
    
    # Config 4: Exponential decay
    n4 = 12
    support4 = np.arange(n4, dtype=np.float64)
    weights4 = np.exp(-0.25 * support4)
    weights4 /= weights4.sum()
    initial_configs.append((support4, weights4))
    
    # Config 5: Concentrated at 0 and 1
    n5 = 5
    support5 = np.array([0, 0.5, 1, 2, 4])
    weights5 = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    weights5 /= weights5.sum()
    initial_configs.append((support5, weights5))
    
    # Config 6: Log-spaced support, larger n
    n6 = 15
    support6 = np.logspace(0, 1.5, n6)  # 1 to ~31
    support6 = np.sort(support6)
    weights6 = np.exp(-0.3 * np.arange(n6))
    weights6 /= weights6.sum()
    initial_configs.append((support6, weights6))
    
    # Config 7: Power-law support
    n7 = 20
    support7 = np.power(np.arange(1, n7+1), 0.7)
    support7 = np.sort(support7)
    weights7 = np.power(np.arange(1, n7+1), -0.8)
    weights7 /= weights7.sum()
    initial_configs.append((support7, weights7))
    
    # Config 8: Very concentrated
    n8 = 4
    support8 = np.array([0, 1, 2, 5])
    weights8 = np.array([0.5, 0.25, 0.15, 0.1])
    weights8 /= weights8.sum()
    initial_configs.append((support8, weights8))
    
    # Config 9: Geometric spacing
    n9 = 10
    support9 = 2.0 ** np.arange(n9) * 0.5  # 0.5, 1, 2, 4, ...
    weights9 = np.array([0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.015, 0.005])
    weights9 /= weights9.sum()
    initial_configs.append((support9, weights9))
    
    # Config 10: Symmetric-ish with gap
    n10 = 7
    support10 = np.array([0, 1, 2, 3, 5, 7, 10])
    weights10 = np.array([0.2, 0.15, 0.12, 0.1, 0.18, 0.15, 0.1])
    weights10 /= weights10.sum()
    initial_configs.append((support10, weights10))
    
    # Evaluate all initial configs
    for support, weights in initial_configs:
        P = _update_best(support, weights)
        if time.time() > deadline - 300:
            break
    
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    # Optimize each initial config
    for support, weights in initial_configs:
        if time.time() > deadline - 60:
            break
        
        # Phase 1: Optimize weights with fixed support
        try:
            s_opt, w_opt, P = _optimize_weights_fixed(support, weights, timeout=80, max_iter=500)
            _update_best(s_opt, w_opt)
        except Exception:
            pass
        
        if time.time() > deadline - 60:
            break
        
        # Phase 2: Joint optimization
        try:
            s_joint, w_joint, P = _optimize_joint(s_opt, w_opt, timeout=80, max_iter=150)
            _update_best(s_joint, w_joint)
        except Exception:
            pass
        
        if time.time() > deadline - 60:
            break
        
        # Phase 3: Local refinement
        try:
            s_ref, w_ref, P = _local_refine(s_joint, w_joint, timeout=60)
            _update_best(s_ref, w_ref)
        except Exception:
            pass
    
    # Additional random restarts from best solution
    if best_support is not None:
        n_best = len(best_support)
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(12345 + restart * 7)
            if n_best <= 10:
                perturbation_scale = 0.2
            else:
                perturbation_scale = 0.1
            
            pert_s = best_support + np.random.randn(n_best) * perturbation_scale * (best_support.max() - best_support.min() + 1)
            pert_s = np.maximum(pert_s, 0.0)
            pert_s = np.sort(pert_s)
            
            pert_w = best_weights * (1 + np.random.randn(n_best) * 0.15)
            pert_w = np.maximum(pert_w, 1e-6)
            pert_w /= pert_w.sum()
            
            _update_best(pert_s, pert_w)
            
            # Quick weight optimization
            try:
                s_opt, w_opt, P = _optimize_weights_fixed(pert_s, pert_w, timeout=40, max_iter=300)
                _update_best(s_opt, w_opt)
            except Exception:
                pass
    
    # Final: try a few more structural variations based on best
    if best_support is not None and time.time() < deadline - 30:
        n_best = len(best_support)
        
        # Try adding a point at 0 with extra weight
        if best_support[0] > 0.01:
            new_s = np.sort(np.concatenate([[0.0], best_support]))
            new_w = np.concatenate([[0.1], best_weights * 0.9])
            _update_best(new_s, new_w)
        
        # Try doubling the spread
        new_s = best_support * 2.0
        _update_best(new_s, best_weights)
        
        # Try compressing
        new_s = best_support * 0.5
        _update_best(new_s, best_weights)
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    # Flatten
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
