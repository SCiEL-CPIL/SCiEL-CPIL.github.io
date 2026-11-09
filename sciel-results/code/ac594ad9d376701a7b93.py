# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] efficiently using sorting and searchsorted.
    
    P = sum_{k,l} w_k * w_l * F(2*s_l - s_k)
    where F(t) = sum_{i,j: s_i + s_j < t} w_i * w_j
    
    We precompute all pairwise sums s_i + s_j with their weights w_i * w_j,
    sort them, then for each (k,l) query use searchsorted.
    """
    n = len(support)
    w = weights / np.sum(weights)
    
    # Build all pairwise sums
    # pairs: (s_i + s_j, w_i * w_j) for all i, j
    # This is O(n^2) space and time
    ss = support[:, None] + support[None, :]  # n x n
    ww = w[:, None] * w[None, :]  # n x n
    
    # Flatten and sort by sum value
    flat_s = ss.ravel()
    flat_w = ww.ravel()
    
    # Sort by sum
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    
    # Cumulative sum of weights for prefix sums
    cum_w = np.cumsum(sorted_w)
    
    # For each (k, l), we need F(2*s_l - s_k) = sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # Thresholds: 2*s_l - s_k for all k, l
    thresholds = 2.0 * support[None, :] - support[:, None]  # n x n, threshold[k,l] = 2*s_l - s_k
    
    # For each threshold, find the count of pairwise sums strictly less than it
    # np.searchsorted(sorted_s, threshold, side='left') gives index of first element >= threshold
    # So the count of elements < threshold is that index
    # But we need sum of weights, not count. So we use the cumulative weight array.
    
    # For each threshold t, idx = searchsorted(sorted_s, t, side='left')
    # Then F(t) = cum_w[idx-1] if idx > 0 else 0
    
    # Vectorize:
    idxs = np.searchsorted(sorted_s, thresholds.ravel(), side='left')
    
    # F(t) = cum_w[idx-1] if idx > 0 else 0
    # We can do: np.where(idxs > 0, cum_w[np.maximum(idxs-1, 0)], 0.0)
    # But need to be careful: when idx=0, we want 0
    F_vals = np.zeros(len(idxs), dtype=np.float64)
    mask = idxs > 0
    F_vals[mask] = cum_w[idxs[mask] - 1]
    
    F_vals = F_vals.reshape(n, n)  # F_vals[k, l] = F(2*s_l - s_k)
    
    # P = sum_{k,l} w_k * w_l * F_vals[k, l]
    P = np.sum(w[:, None] * w[None, :] * F_vals)
    
    return P

def _neg_P(params):
    """Negative P for minimization. params = [r, alpha] for geometric support/weights."""
    r, alpha = params
    if r <= 0 or r >= 1.0 or alpha <= 0 or alpha >= 1.0:
        return 1e10  # penalty for invalid
    
    n = 32
    support = r ** np.arange(n, dtype=np.float64)
    weights = alpha ** np.arange(n, dtype=np.float64)
    
    # Check validity
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1e10
    if np.sum(weights) <= 1e-9:
        return 1e10
    
    try:
        P = _compute_P_fast(support, weights)
    except:
        return 1e10
    
    return -P

def _neg_P_general(params):
    """Negative P for general n-point optimization. params = [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]."""
    n = len(params) // 2
    support = params[:n]
    weights = params[n:]
    
    # Enforce non-negativity
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1e10
    if np.sum(weights) <= 1e-9:
        return 1e10
    
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)
    
    try:
        P = _compute_P_fast(support, weights)
    except:
        return 1e10
    
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. First optimize over 2-parameter geometric family (r, alpha) with n=32.
    2. Then use the best result as initialization for a general n-point optimization.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    deadline = time.time() + 580  # 580 seconds, leaving 20s safety margin
    
    best_P = -1e10
    best_support = None
    best_weights = None
    
    # Phase 1: Optimize over geometric family (r, alpha) with n=32
    n_geo = 32
    
    # Use differential evolution for global search over (r, alpha)
    bounds = [(1e-6, 0.9999), (1e-6, 0.9999)]
    
    try:
        result_de = differential_evolution(
            _neg_P, bounds, seed=42, maxiter=50, popsize=15,
            tol=1e-6, polish=True
        )
        if result_de.success and result_de.fun < 0:
            P_de = -result_de.fun
            r_opt, alpha_opt = result_de.x
            support = r_opt ** np.arange(n_geo, dtype=np.float64)
            weights = alpha_opt ** np.arange(n_geo, dtype=np.float64)
            if P_de > best_P:
                best_P = P_de
                best_support = support.copy()
                best_weights = weights.copy()
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if best_support is not None:
        try:
            x0 = [best_support[1]/best_support[0] if best_support[0] > 1e-10 else 0.5,
                  best_weights[1]/best_weights[0] if best_weights[0] > 1e-10 else 0.5]
            result_nm = minimize(_neg_P, x0, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if result_nm.fun < 0:
                P_nm = -result_nm.fun
                r_opt, alpha_opt = result_nm.x
                support = r_opt ** np.arange(n_geo, dtype=np.float64)
                weights = alpha_opt ** np.arange(n_geo, dtype=np.float64)
                if P_nm > best_P:
                    best_P = P_nm
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass
    
    # Phase 2: General n-point optimization starting from best geometric result
    if best_support is not None and time.time() < deadline - 100:
        n_gen = min(12, len(best_support))  # Use smaller n for general optimization
        # Take first n_gen points from best support/weights
        s0 = best_support[:n_gen].copy()
        w0 = best_weights[:n_gen].copy()
        
        # Normalize weights
        w0 = w0 / np.sum(w0)
        
        # Initialize general params: [s_0,...,s_{n-1}, w_0,...,w_{n-1}]
        params0 = np.concatenate([s0, w0])
        
        # Bounds for general optimization
        # Support: [0, 100] (arbitrary upper bound)
        # Weights: [0, 1]
        bounds_gen = [(0, 100)] * n_gen + [(0, 1)] * n_gen
        
        # Use Powell's method for local optimization
        try:
            result_powell = minimize(
                _neg_P_general, params0, method='Powell',
                bounds=bounds_gen,
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if result_powell.fun < 0:
                P_powell = -result_powell.fun
                s_opt = np.maximum(result_powell.x[:n_gen], 0.0)
                w_opt = np.maximum(result_powell.x[n_gen:], 0.0)
                if np.sum(w_opt) > 1e-9:
                    if P_powell > best_P:
                        best_P = P_powell
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead as well
        if time.time() < deadline - 50:
            try:
                result_nm2 = minimize(
                    _neg_P_general, params0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if result_nm2.fun < 0:
                    P_nm2 = -result_nm2.fun
                    s_opt = np.maximum(result_nm2.x[:n_gen], 0.0)
                    w_opt = np.maximum(result_nm2.x[n_gen:], 0.0)
                    if np.sum(w_opt) > 1e-9:
                        if P_nm2 > best_P:
                            best_P = P_nm2
                            best_support = s_opt.copy()
                            best_weights = w_opt.copy()
            except Exception:
                pass
    
    # Fallback: if nothing worked, use uniform on {0, 1, ..., 31}
    if best_support is None or best_P <= 0:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure validity
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if np.sum(best_weights) <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
    
    # Final check and return
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert np.all(np.isfinite(best_support))
    assert np.all(np.isfinite(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
