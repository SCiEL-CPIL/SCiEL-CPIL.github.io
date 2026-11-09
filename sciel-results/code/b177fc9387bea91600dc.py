# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys


def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums: s_i + s_j for all i, j
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # shape (n, n)
    pair_probs = np.outer(w, w)  # shape (n, n)
    
    # For each (k, l): threshold = 2*s_l - s_k
    # We need sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < threshold)
    # Then sum over k, l: w_k * w_l * (above)
    
    # Vectorized: for each k, l, compute the inner sum
    # threshold matrix: shape (n, n) where threshold[k, l] = 2*s_l - s_k
    s_k = s.reshape(-1, 1)  # (n, 1)
    s_l = s.reshape(1, -1)  # (1, n)
    thresholds = 2.0 * s_l - s_k  # (n, n)
    
    # For each (k, l), we need sum over (i, j) of pair_probs[i,j] * I(pair_sums[i,j] < thresholds[k,l])
    # This is expensive: O(n^4) naive. Let's vectorize over k, l.
    
    # pair_sums shape (n, n), thresholds shape (n, n)
    # We need: for each (k,l), sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < thresholds[k,l])
    
    # Approach: for each k, l, compute the inner sum. This is O(n^2) per (k,l), total O(n^4).
    # For n up to ~30-40, this is manageable.
    
    # More vectorized: 
    # inner_sum[k, l] = sum_{i,j} pair_probs[i,j] * (pair_sums[i,j] < thresholds[k,l])
    # = sum_{i,j} pair_probs[i,j] * (pair_sums[i,j] < 2*s_l - s_k)
    
    # Let's do it with broadcasting:
    # pair_sums: (n, n) -> (1, 1, n, n)
    # thresholds: (n, n) -> (n, n, 1, 1)
    # indicator: (n, n, n, n)
    # inner_sum[k, l] = sum over i, j of pair_probs[i,j] * indicator[k, l, i, j]
    
    ps = pair_sums.reshape(1, 1, n, n)  # (1, 1, n, n)
    th = thresholds.reshape(n, n, 1, 1)  # (n, n, 1, 1)
    indicator = (ps < th).astype(np.float64)  # (n, n, n, n)
    
    # inner_sum[k, l] = sum_{i,j} pair_probs[i,j] * indicator[k,l,i,j]
    # pair_probs: (n, n) -> (1, 1, n, n)
    pp = pair_probs.reshape(1, 1, n, n)
    inner_sum = np.sum(pp * indicator, axis=(2, 3))  # (n, n)
    
    # Final: sum_{k,l} w_k * w_l * inner_sum[k, l]
    wl = np.outer(w, w)  # (n, n)
    P = np.sum(wl * inner_sum)
    
    return P


def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] using a more memory-efficient approach.
    For moderate n, this should be fast enough.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Sort support and weights together for potential optimization
    order = np.argsort(s)
    s = s[order]
    w = w[order]
    
    # Pair sums and probabilities
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    pair_probs = np.outer(w, w)  # (n, n)
    
    # For each (k, l), compute inner sum
    # threshold[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]  # (n, n)
    
    # Compute inner_sum[k, l] = sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < thresholds[k,l])
    # Using broadcasting
    ps = pair_sums[None, None, :, :]  # (1, 1, n, n)
    th = thresholds[:, :, None, None]  # (n, n, 1, 1)
    indicator = (ps < th).astype(np.float64)  # (n, n, n, n)
    
    pp = pair_probs[None, None, :, :]  # (1, 1, n, n)
    inner_sum = np.sum(pp * indicator, axis=(2, 3))  # (n, n)
    
    wl = np.outer(w, w)
    P = np.sum(wl * inner_sum)
    
    return P


def _neg_P(params, n):
    """
    Objective function to minimize (negative of P).
    params: flattened array of [support_0, ..., support_{n-1}, weight_0, ..., weight_{n-1}]
    """
    s = params[:n]
    w = params[n:]
    
    # Ensure support is non-negative
    s = np.maximum(s, 0.0)
    # Ensure weights are positive
    w = np.maximum(w, 1e-12)
    
    return -_compute_P_fast(s, w)


def _optimize_with_scipy(n, max_time=500.0):
    """
    Optimize using scipy's differential_evolution and then polish with Nelder-Mead.
    """
    start_time = time.time()
    
    # Define bounds
    # Support: [0, 100]
    # Weights: [1e-6, 1]
    bounds_s = [(0.0, 100.0)] * n
    bounds_w = [(1e-6, 1.0)] * n
    bounds = bounds_s + bounds_w
    
    # Initial guess
    x0 = np.concatenate([
        np.linspace(0.1, 10.0, n),
        np.ones(n) / n
    ])
    
    # Differential evolution
    result_de = differential_evolution(
        _neg_P, bounds, args=(n,),
        maxiter=200, popsize=15, tol=1e-8,
        seed=42, polish=True, init='sobol'
    )
    
    elapsed = time.time() - start_time
    if elapsed > max_time - 60:
        # Not enough time for polishing
        best_params = result_de.x
        best_s = np.maximum(best_params[:n], 0.0)
        best_w = np.maximum(best_params[n:], 1e-12)
        return best_s, best_w
    
    # Polish with Nelder-Mead
    result_nm = minimize(
        _neg_P, result_de.x, args=(n,),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
    )
    
    best_params = result_nm.x
    best_s = np.maximum(best_params[:n], 0.0)
    best_w = np.maximum(best_params[n:], 1e-12)
    
    return best_s, best_w


def _try_small_n(n, deadline):
    """
    Try to optimize for a given small n.
    """
    try:
        s, w = _optimize_with_scipy(n, max_time=deadline)
        P = _compute_P_fast(s, w)
        return s, w, P
    except Exception as e:
        return None, None, 0.0


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin before 630s limit
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Baseline: uniform on {0, 1, ..., 31}
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    P_base = _compute_P_fast(support_base, weights_base)
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    best_P = P_base
    
    # Try small n values with optimization
    for n in [2, 3, 4, 5, 6]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 100:
            break
        
        s, w, P = _try_small_n(n, deadline - elapsed)
        if s is not None and P > best_P:
            best_support = s
            best_weights = w
            best_P = P
    
    # Try n=8 with more time if available
    elapsed = time.time() - start_time
    if elapsed < deadline - 150:
        s, w, P = _try_small_n(8, deadline - elapsed)
        if s is not None and P > best_P:
            best_support = s
            best_weights = w
            best_P = P
    
    # Final validation
    if best_support is None or best_weights is None:
        return support_base, weights_base
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return support_base, weights_base
    
    # Ensure support >= 0
    best_support = np.maximum(best_support, 0.0)
    # Ensure weights >= 0
    best_weights = np.maximum(best_weights, 1e-12)
    
    # Ensure sum of weights > 1e-9
    if np.sum(best_weights) <= 1e-9:
        return support_base, weights_base
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
