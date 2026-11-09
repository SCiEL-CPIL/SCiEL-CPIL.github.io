# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 620.0  # safety margin before 630s kill


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    n = len(support)
    # Normalize weights
    w = weights.astype(np.float64)
    s = support.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    
    # Pairwise sums: s_i + s_j for all i,j
    pair_sums = s[:, None] + s[None, :]  # n x n
    pair_weights = w[:, None] * w[None, :]  # n x n
    
    # For each (k, l), we need sum of w_i*w_j over (i,j) with s_i+s_j < 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Count weighted pairs with pair_sums[i,j] < threshold
            mask = pair_sums < threshold
            inner = np.sum(pair_weights * mask)
            P += w[k] * w[l] * inner
    return P


def _compute_P_fast(support, weights):
    """Faster computation using sorting."""
    n = len(support)
    w = weights.astype(np.float64)
    s = support.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    
    # All pairwise sums
    pair_sums = (s[:, None] + s[None, :]).ravel()
    pair_weights = (w[:, None] * w[None, :]).ravel()
    
    # Sort by pair_sums
    idx = np.argsort(pair_sums)
    sorted_sums = pair_sums[idx]
    sorted_w = pair_weights[idx]
    cum_w = np.cumsum(sorted_w)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of pairs with sum < threshold
            count = np.searchsorted(sorted_sums, threshold, side='left')
            inner = cum_w[count - 1] if count > 0 else 0.0
            P += w[k] * w[l] * inner
    return P


def _neg_P(params, n_support):
    """Objective for optimization: negative P."""
    # params: [support_0, ..., support_{n-1}, weight_0, ..., weight_{n-1}]
    support = params[:n_support]
    weights = params[n_support:]
    
    # Ensure support >= 0 and weights >= 0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1e10
    if np.sum(weights) <= 1e-9:
        return 1e10
    
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)
    
    return -_compute_P_fast(support, weights)


def _try_n_points(n, time_budget):
    """Try to optimize an n-point distribution."""
    start_time = time.time()
    best_P = 0.0
    best_params = None
    
    # Initial guesses
    inits = []
    
    # Uniform on [0, n-1]
    sup0 = np.arange(n, dtype=np.float64)
    w0 = np.ones(n, dtype=np.float64) / n
    inits.append(np.concatenate([sup0, w0]))
    
    # 2-point extended: mass at 0 and some values
    for b in [1.0, 2.0, 5.0, 10.0]:
        sup = np.zeros(n)
        sup[-1] = b
        w = np.zeros(n)
        w[0] = 0.5
        w[-1] = 0.5
        inits.append(np.concatenate([sup, w]))
    
    # Spread out
    sup = np.linspace(0, 10, n)
    w = np.ones(n) / n
    inits.append(np.concatenate([sup, w]))
    
    # Geometric spread
    sup = np.logspace(-3, 1, n)
    sup = np.maximum(sup, 0)
    w = np.ones(n) / n
    inits.append(np.concatenate([sup, w]))
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        
        try:
            result = minimize(
                _neg_P, init, args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return None, 0.0
    
    support = np.maximum(best_params[:n], 0.0)
    weights = np.maximum(best_params[n:], 0.0)
    return support, weights, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Try small n first, then larger
    for n in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        
        time_budget = min(60.0, deadline - time.time() - 10)
        if time_budget < 5:
            break
        
        try:
            support, weights, P_val = _try_n_points(n, time_budget)
            if support is not None and P_val > best_P:
                best_P = P_val
                best_support = support.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # If nothing worked, return baseline
    if best_support is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    # Final check: ensure validity
    if len(best_support) != len(best_weights) or len(best_support) < 1:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    if np.sum(best_weights) <= 1e-9:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    # Clamp small negatives
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
