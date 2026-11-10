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
    
    # Precompute all pairwise sums and their probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Flatten and sort pair sums for efficient lookup
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum value
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities for fast prefix queries
    cum_probs = np.cumsum(sorted_probs)
    
    # Vectorized computation over (k, l) pairs
    # thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find count of pair sums strictly less
    # searchsorted with side='left' gives index of first element >= threshold
    # So count of elements < threshold is that index
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    # Get cumulative probabilities at those indices (handle index 0)
    # If count is 0, inner_sum is 0; otherwise cum_probs[count-1]
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    # P = sum over k,l of w_k * w_l * inner_sum[k,l]
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    
    return float(P)

def _grid_search_3point(deadline):
    """Grid search over 3-point distributions {0, a, b}."""
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # Grid for a and b
    a_vals = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=np.float64)
    b_vals = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0], dtype=np.float64)
    
    # Weight grid
    w1_vals = np.arange(0.0, 1.01, 0.05, dtype=np.float64)
    
    for a in a_vals:
        for b in b_vals:
            if b <= a:
                continue
            for w1 in w1_vals:
                for w2 in np.arange(0.0, 1.0 - w1 + 1e-9, 0.05, dtype=np.float64):
                    w0 = 1.0 - w1 - w2
                    if w0 < -1e-9:
                        continue
                    w0 = max(w0, 0.0)
                    
                    support = np.array([0.0, a, b], dtype=np.float64)
                    weights = np.array([w0, w1, w2], dtype=np.float64)
                    
                    P = _compute_P(support, weights)
                    if P > best_P:
                        best_P = P
                        best_support = support.copy()
                        best_weights = weights.copy()
        
        if time.time() > deadline - 30:
            break
        if time.time() > deadline:
            break
    
    return best_support, best_weights, best_P

def _grid_search_4point(deadline, init_support=None, init_weights=None):
    """Grid search over 4-point distributions {0, a, b, c}."""
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # If we have an initial 3-point solution, expand around it
    if init_support is not None and len(init_support) >= 3:
        a0, b0 = init_support[1], init_support[2]
        a_vals = np.array([max(0.05, a0 * 0.5), a0 * 0.75, a0, a0 * 1.25, a0 * 1.5, a0 * 2.0], dtype=np.float64)
        b_vals = np.array([max(a0 * 1.1, b0 * 0.5), b0 * 0.75, b0, b0 * 1.25, b0 * 1.5, b0 * 2.0], dtype=np.float64)
    else:
        a_vals = np.array([0.2, 0.5, 1.0, 2.0], dtype=np.float64)
        b_vals = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    
    c_vals = np.array([2.0, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0], dtype=np.float64)
    
    w1_vals = np.arange(0.0, 1.01, 0.1, dtype=np.float64)
    
    for a in a_vals:
        for b in b_vals:
            if b <= a:
                continue
            for c in c_vals:
                if c <= b:
                    continue
                for w1 in w1_vals:
                    for w2 in np.arange(0.0, 1.0 - w1 + 1e-9, 0.1, dtype=np.float64):
                        for w3 in np.arange(0.0, 1.0 - w1 - w2 + 1e-9, 0.1, dtype=np.float64):
                            w0 = 1.0 - w1 - w2 - w3
                            if w0 < -1e-9:
                                continue
                            w0 = max(w0, 0.0)
                            
                            support = np.array([0.0, a, b, c], dtype=np.float64)
                            weights = np.array([w0, w1, w2, w3], dtype=np.float64)
                            
                            P = _compute_P(support, weights)
                            if P > best_P:
                                best_P = P
                                best_support = support.copy()
                                best_weights = weights.copy
                        
                        if time.time() > deadline - 15:
                            break
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break
        if time.time() > deadline:
            break
    
    return best_support, best_weights, best_P

def _local_refine(support, weights, deadline):
    """Local refinement using Nelder-Mead on log-weights and support."""
    n = len(support)
    
    def _neg_P(params):
        log_w = params[:n]
        s = params[n:]
        w = np.exp(log_w - np.max(log_w))
        s = np.maximum(s, 0.0)
        s = np.sort(s)
        return -_compute_P(s, w)
    
    init_log_w = np.log(weights + 1e-15)
    init_s = np.sort(support)
    init_params = np.concatenate([init_log_w, init_s])
    
    best_params = init_params.copy()
    best_val = _neg_P(best_params)
    
    try:
        res = minimize(_neg_P, best_params, method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    log_w = best_params[:n]
    s = np.maximum(best_params[n:], 0.0)
    s = np.sort(s)
    w = np.exp(log_w - np.max(log_w))
    
    return s, w, -best_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Start with a known good configuration
    s_init = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    w_init = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    P_init = _compute_P(s_init, w_init)
    _update_best(s_init, w_init, P_init)
    
    # Grid search over 3-point distributions
    if time.time() < deadline - 60:
        s3, w3, P3 = _grid_search_3point(deadline)
        if s3 is not None:
            _update_best(s3, w3, P3)
    
    # Local refinement on best 3-point
    if best_support is not None and time.time() < deadline - 30:
        s_ref, w_ref, P_ref = _local_refine(best_support, best_weights, deadline)
        _update_best(s_ref, w_ref, P_ref)
    
    # Grid search over 4-point distributions, initialized from best 3-point
    if time.time() < deadline - 15:
        s4, w4, P4 = _grid_search_4point(deadline, best_support, best_weights)
        if s4 is not None:
            _update_best(s4, w4, P4)
    
    # Final local refinement
    if best_support is not None and time.time() < deadline - 5:
        s_ref, w_ref, P_ref = _local_refine(best_support, best_weights, deadline)
        _update_best(s_ref, w_ref, P_ref)
    
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
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
