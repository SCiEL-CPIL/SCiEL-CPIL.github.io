# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

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
    
    # Precompute all pairwise sums and their probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Flatten and sort pair sums for efficient lookup
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum value
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities for fast prefix queries
    cum_probs = np.cumsum(sorted_probs)
    
    # Vectorized computation over (k, l) pairs
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find count of pair sums strictly less
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    
    return float(P)

def _compute_P_fast(s_sorted, w, sorted_sums, sort_order):
    """Fast P computation with precomputed sorted pair sums."""
    n = len(s_sorted)
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    pair_probs = w[:, None] * w[None, :]
    flat_probs = pair_probs.flatten()
    sorted_probs = flat_probs[sort_order]
    cum_probs = np.cumsum(sorted_probs)
    
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w[:, None] * w[None, :] * inner_sums)
    return float(P)

def _precompute_support(support):
    """Precompute sorted pair sums for a fixed support."""
    s = np.asarray(support, dtype=np.float64)
    order = np.argsort(s)
    s_sorted = s[order]
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    flat_sums = pair_sums.flatten()
    sort_order = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_order]
    return s_sorted, sorted_sums, sort_order

def _coordinate_ascent(s_sorted, w_init, sorted_sums, sort_order, max_sweeps=50, tol=1e-10, deadline=None):
    """Coordinate ascent on simplex to maximize P for fixed support."""
    n = len(s_sorted)
    w = w_init.copy().astype(np.float64)
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum <= 1e-9:
        w = np.ones(n) / n
    else:
        w = w / w_sum
    
    best_P = _compute_P_fast(s_sorted, w, sorted_sums, sort_order)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Optimize w_i with others fixed
            # P is a quartic in w_i: P = a*w_i^4 + b*w_i^3 + c*w_i^2 + d*w_i + e
            # where coefficients depend on other weights
            
            # Save current
            w_old_i = w[i]
            w_rest = w.copy()
            w_rest[i] = 0.0
            S_rest = w_rest.sum()
            
            if S_rest >= 1.0 - 1e-15:
                # All weight on i
                w[i] = 1.0
                for j in range(n):
                    if j != i:
                        w[j] = 0.0
                P_new = _compute_P_fast(s_sorted, w, sorted_sums, sort_order)
                if P_new > best_P:
                    best_P = P_new
                    improved = True
                continue
            
            # w_i ranges from 0 to 1-S_rest (but w_rest sums to S_rest, so we need to scale)
            # Actually, let's parameterize: w_i = t, and scale others by (1-t)/S_rest
            # P(t) = sum over all terms involving w_i
            
            # Simpler: try several candidate values
            candidates = [0.0, 1.0 - S_rest, w_old_i, (1.0 - S_rest) * 0.25, (1.0 - S_rest) * 0.5, (1.0 - S_rest) * 0.75]
            
            best_t = w_old_i
            best_P_i = -1.0
            
            for t in candidates:
                t = max(0.0, min(1.0 - S_rest, t))
                w_test = w_rest * (1.0 - t) / S_rest
                w_test[i] = t
                P_test = _compute_P_fast(s_sorted, w_test, sorted_sums, sort_order)
                if P_test > best_P_i:
                    best_P_i = P_test
                    best_t = t
            
            # Refine around best_t
            if best_P_i > best_P:
                # Golden section search around best_t
                lo = max(0.0, best_t - (1.0 - S_rest) * 0.1)
                hi = min(1.0 - S_rest, best_t + (1.0 - S_rest) * 0.1)
                if hi > lo:
                    gr = (np.sqrt(5) - 1) / 2
                    a1 = lo + (1 - gr) * (hi - lo)
                    b1 = lo + gr * (hi - lo)
                    
                    w_a = w_rest * (1.0 - a1) / S_rest
                    w_a[i] = a1
                    P_a = _compute_P_fast(s_sorted, w_a, sorted_sums, sort_order)
                    
                    w_b = w_rest * (1.0 - b1) / S_rest
                    w_b[i] = b1
                    P_b = _compute_P_fast(s_sorted, w_b, sorted_sums, sort_order)
                    
                    for _ in range(30):
                        if deadline is not None and time.time() > deadline:
                            break
                        if P_a < P_b:
                            lo = a1
                            a1 = b1
                            P_a = P_b
                            b1 = lo + gr * (hi - lo)
                            w_b = w_rest * (1.0 - b1) / S_rest
                            w_b[i] = b1
                            P_b = _compute_P_fast(s_sorted, w_b, sorted_sums, sort_order)
                        else:
                            hi = b1
                            b1 = a1
                            P_b = P_a
                            a1 = lo + (1 - gr) * (hi - lo)
                            w_a = w_rest * (1.0 - a1) / S_rest
                            w_a[i] = a1
                            P_a = _compute_P_fast(s_sorted, w_a, sorted_sums, sort_order)
                    
                    best_t = (lo + hi) / 2
                    w_best = w_rest * (1.0 - best_t) / S_rest
                    w_best[i] = best_t
                    P_best = _compute_P_fast(s_sorted, w_best, sorted_sums, sort_order)
                    
                    if P_best > best_P:
                        w = w_best
                        best_P = P_best
                        improved = True
        
        if not improved:
            break
    
    return w, best_P

def _optimize_grid(grid_size, deadline, max_sweeps=30):
    """Optimize weights on a grid of support points using coordinate ascent."""
    s = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    s_sorted, sorted_sums, sort_order = _precompute_support(s)
    
    # Multiple initializations
    best_w = None
    best_P = -1.0
    
    inits = [
        np.ones(grid_size) / grid_size,
        np.exp(-2.0 * s) / np.exp(-2.0 * s).sum(),
        np.exp(-4.0 * s) / np.exp(-4.0 * s).sum(),
        np.concatenate([np.ones(grid_size//2)*0.7, np.ones(grid_size-grid_size//2)*0.3]) / grid_size,
        np.array([1.0 if i < 3 else 0.01 for i in range(grid_size)]) / (3.0 + 0.01*(grid_size-3)),
    ]
    
    for w_init in inits:
        if time.time() > deadline:
            break
        w, P = _coordinate_ascent(s_sorted, w_init, sorted_sums, sort_order, 
                                    max_sweeps=max_sweeps, deadline=deadline)
        if P > best_P:
            best_P = P
            best_w = w
    
    return s, best_w, best_P

def _neg_P_w(params, s_sorted, sorted_sums, sort_order):
    """Negative P for weight optimization with fixed support."""
    log_w = params
    w = np.exp(log_w - np.max(log_w))
    return -_compute_P_fast(s_sorted, w, sorted_sums, sort_order)

def _optimize_fixed_support(initial_support, initial_weights, max_iter=300, timeout=60):
    """Optimize weights given fixed support points using precomputed structure."""
    n = len(initial_support)
    s_sorted, sorted_sums, sort_order = _precompute_support(initial_support)
    
    init_log_w = np.log(initial_weights + 1e-15)
    best_x = init_log_w.copy()
    best_val = _neg_P_w(best_x, s_sorted, sorted_sums, sort_order)
    
    start_time = time.time()
    
    try:
        res = minimize(_neg_P_w, best_x, args=(s_sorted, sorted_sums, sort_order),
                     method='Powell', options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time < timeout:
        try:
            res = minimize(_neg_P_w, best_x, args=(s_sorted, sorted_sums, sort_order),
                         method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    w = np.exp(best_x - np.max(best_x))
    return initial_support, w, -best_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 580  # Safety margin
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Phase 1: Grid-based coordinate ascent
    for grid_size in [32, 64, 128]:
        if time.time() > deadline - 100:
            break
        try:
            s, w, P = _optimize_grid(grid_size, deadline - 50, max_sweeps=20)
            if s is not None and w is not None and P > best_P:
                _update_best(s, w, P)
        except Exception:
            pass
    
    # Phase 2: Small support local optimization (for refinement)
    if time.time() < deadline - 60:
        initial_configs = []
        
        for n in [3, 4, 5, 6, 8]:
            s = np.linspace(0, 1, n, dtype=np.float64)
            w = np.ones(n, dtype=np.float64) / n
            initial_configs.append((s, w))
            
            s = np.linspace(0, 1, n, dtype=np.float64)
            w = np.exp(-3.0 * s)
            w = w / w.sum()
            initial_configs.append((s, w))
            
            if n >= 4:
                half = n // 2
                s = np.concatenate([np.linspace(0, 0.2, half), np.linspace(0.5, 1.0, n - half)])
                w = np.ones(n, dtype=np.float64) / n
                initial_configs.append((s, w))
        
        s = np.array([0.0, 0.05, 0.1, 0.3, 0.7, 1.0])
        w = np.array([0.35, 0.2, 0.15, 0.1, 0.1, 0.1])
        initial_configs.append((s, w))
        
        s = np.array([0.0, 0.1, 0.5, 1.0])
        w = np.array([0.5, 0.2, 0.15, 0.15])
        initial_configs.append((s, w))
        
        for support, weights in initial_configs:
            if time.time() > deadline - 30:
                break
            P = _compute_P(support, weights)
            _update_best(support, weights, P)
        
        # Optimize top small configs
        config_scores = []
        for support, weights in initial_configs:
            P = _compute_P(support, weights)
            config_scores.append((P, support, weights))
        config_scores.sort(key=lambda x: -x[0])
        
        for P_init, support, weights in config_scores[:4]:
            if time.time() > deadline - 20:
                break
            try:
                opt_support, opt_weights, P = _optimize_fixed_support(support, weights, max_iter=200, timeout=30)
                _update_best(opt_support, opt_weights, P)
            except Exception:
                pass
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Clean up: ensure no NaN/inf, non-negative
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights is positive
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    # Remove zero-weight points to keep solution sparse
    mask = best_weights > 1e-10
    if mask.sum() > 0 and mask.sum() < len(best_weights):
        best_support = best_support[mask]
        best_weights = best_weights[mask]
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
