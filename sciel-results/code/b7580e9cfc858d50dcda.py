# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 580  # 580s deadline, 50s safety margin


def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    P = sum_{k,l} w_k * w_l * C(2*s_l - s_k)
    where C(t) = sum_{i,j} w_i * w_j * I(s_i + s_j < t)
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums and probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w)  # (n, n)
    
    # Sort pair_sums for efficient CDF lookup
    ps_flat = pair_sums.ravel()
    pp_flat = pair_probs.ravel()
    order = np.argsort(ps_flat)
    ps_sorted = ps_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.concatenate([[0.0], np.cumsum(pp_sorted)])
    
    # For each (k, l): threshold = 2*s_l - s_k
    # C(threshold) = pp_cumsum[searchsorted(ps_sorted, threshold, side='left')]
    
    # Vectorize over (k, l)
    kk, ll = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    kk = kk.ravel()
    ll = ll.ravel()
    
    thresholds = 2.0 * s[ll] - s[kk]  # shape (n^2,)
    idx = np.searchsorted(ps_sorted, thresholds, side='left')
    C_vals = pp_cumsum[idx]  # shape (n^2,)
    
    # P = sum_{k,l} w_k * w_l * C_vals
    P = np.sum(w[kk] * w[ll] * C_vals)
    
    return float(P)


def _neg_P_params(params, n):
    """
    Objective for differential_evolution.
    params = [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]
    We use softplus for support (ensures > 0) and softmax for weights (ensures positive, sums to 1).
    Actually, let's just use raw values with bounds and normalize.
    """
    s_raw = params[:n]
    w_raw = params[n:]
    
    # Ensure support is non-negative
    s = np.maximum(s_raw, 0.0)
    
    # Ensure weights are positive and normalize
    w = np.maximum(w_raw, 1e-10)
    w = w / np.sum(w)
    
    P = _compute_P_fast(s, w)
    return -P


def _optimize_n_points(n, max_iter=1000, tol=1e-6, popsize=15, timeout=60):
    """
    Use differential_evolution to optimize n-point distribution.
    """
    start_time = time.time()
    
    # Bounds: support in [0, 50], weights in [0.01, 1]
    # Actually, for differential_evolution, we need bounds for each parameter.
    # Let's use a transformation: s_i in [0, 20], w_i in [0.01, 1]
    # But weights should sum to something, let's just let DE handle it and normalize.
    bounds = [(0.0, 20.0)] * n + [(0.01, 1.0)] * n
    
    # Initial population
    # Try a few good initial guesses
    inits = []
    # Two-point-like: {0, L}
    if n >= 2:
        s_init = np.zeros(n)
        w_init = np.ones(n)
        s_init[0] = 0.0
        s_init[1] = 5.0
        w_init[0] = 0.6
        w_init[1] = 0.4
        for i in range(2, n):
            s_init[i] = 5.0 + i
            w_init[i] = 0.1 / i
        inits.append(np.concatenate([s_init, w_init]))
    
    # Uniform-like
    s_init = np.linspace(0, 10, n)
    w_init = np.ones(n) / n
    inits.append(np.concatenate([s_init, w_init]))
    
    # Geometric
    s_init = np.array([0.0] + [2.0**i for i in range(1, n)])
    w_init = np.array([0.7**i for i in range(n)])
    w_init = w_init / w_init.sum()
    inits.append(np.concatenate([s_init, w_init]))
    
    try:
        result = differential_evolution(
            _neg_P_params, bounds, args=(n,),
            maxiter=max_iter, tol=tol, popsize=popsize,
            seed=42, polish=True
        )
    except Exception as e:
        return None, None, 0.0
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return None, None, 0.0
    
    x_opt = result.x
    s_opt = np.maximum(x_opt[:n], 0.0)
    w_opt = np.maximum(x_opt[n:], 1e-10)
    w_opt = w_opt / np.sum(w_opt)
    
    P_opt = _compute_P_fast(s_opt, w_opt)
    
    return s_opt, w_opt, P_opt


def _refine_nelder_mead(support, weights, max_iter=2000, timeout=30):
    """
    Refine a solution using Nelder-Mead.
    """
    from scipy.optimize import minimize
    
    n = len(support)
    
    def _neg_P(x):
        s = np.maximum(x[:n], 0.0)
        w = np.maximum(x[n:], 1e-10)
        w = w / np.sum(w)
        return -_compute_P_fast(s, w)
    
    x0 = np.concatenate([support, weights])
    
    start_time = time.time()
    
    try:
        result = minimize(
            _neg_P, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
    except Exception:
        return support, weights, _compute_P_fast(support, weights)
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return support, weights, _compute_P_fast(support, weights)
    
    x_opt = result.x
    s_opt = np.maximum(x_opt[:n], 0.0)
    w_opt = np.maximum(x_opt[n:], 1e-10)
    w_opt = w_opt / np.sum(w_opt)
    
    P_opt = _compute_P_fast(s_opt, w_opt)
    P_init = _compute_P_fast(support, weights)
    
    if P_opt > P_init:
        return s_opt, w_opt, P_opt
    else:
        return support, weights, P_init


def _try_analytic_two_point():
    """
    Two-point distribution {0, L} with weight p at 0.
    P = (1-p)*p^2*(3-2p), independent of L.
    Optimal p = (15-sqrt(33))/16
    """
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s = np.array([0.0, 1.0])
    w = np.array([p_opt, 1.0 - p_opt])
    P = _compute_P_fast(s, w)
    return s, w, P


def _try_sparse_gap(n, gap_ratio=10.0):
    """
    Try distributions with a large gap between small and large support points.
    This exploits the strict inequality: when X4 is large, most sums are < 2*X4.
    """
    best_P = -1.0
    best_s = None
    best_w = None
    
    # Try different gap ratios and number of small points
    for n_small in range(1, min(n, 4)):
        n_large = n - n_small
        if n_large < 1:
            continue
        
        for gap in [5.0, 10.0, 20.0, 50.0]:
            # Small points near 0
            s_small = np.linspace(0.0, 1.0, n_small)
            # Large points
            s_large = np.linspace(gap, gap + 1.0, n_large)
            s = np.concatenate([s_small, s_large])
            
            # Try different weight distributions
            for p_small in [0.3, 0.5, 0.7, 0.8]:
                w_small = np.ones(n_small) * p_small / n_small
                w_large = np.ones(n_large) * (1.0 - p_small) / n_large
                w = np.concatenate([w_small, w_large])
                
                P = _compute_P_fast(s, w)
                if P > best_P:
                    best_P = P
                    best_s = s.copy()
                    best_w = w.copy()
    
    return best_s, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    global _DEADLINE
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # 1. Analytic two-point solution
    if time.time() < _DEADLINE:
        s2, w2, P2 = _try_analytic_two_point()
        best_support = s2.copy()
        best_weights = w2.copy()
        best_P = P2
    
    # 2. Sparse gap heuristic
    if time.time() < _DEADLINE:
        for n in [3, 4, 5, 6]:
            s, w, P = _try_sparse_gap(n)
            if s is not None and P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
    
    # 3. Differential evolution for n=3, 4, 5
    for n in [3, 4, 5]:
        if time.time() > _DEADLINE - 60:
            break
        s, w, P = _optimize_n_points(n, max_iter=500, popsize=15, timeout=45)
        if s is not None and P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # 4. Nelder-Mead refinement on best found
    if time.time() < _DEADLINE - 30:
        n = len(best_support)
        s_opt, w_opt, P_opt = _refine_nelder_mead(
            best_support, best_weights, max_iter=5000, timeout=40
        )
        if P_opt > best_P:
            best_P = P_opt
            best_support = s_opt.copy()
            best_weights = w_opt.copy()
    
    # 5. Final verification and safety checks
    P_final = _compute_P_fast(best_support, best_weights)
    
    # Ensure validity
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert not np.any(np.isnan(best_support))
    assert not np.any(np.isinf(best_support))
    assert not np.any(np.isnan(best_weights))
    assert not np.any(np.isinf(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    # Clean up: ensure no negative values due to numerical issues
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights to sum to 1
    w_sum = np.sum(best_weights)
    if w_sum > 0:
        best_weights = best_weights / w_sum
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
