# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    n = len(support)
    # Normalize weights
    w = weights / weights.sum()
    s = support
    
    # Build pairwise sums: s_i + s_j
    s_i = s[:, None]
    s_j = s[None, :]
    pair_sums = s_i + s_j  # shape (n, n)
    
    # For each (k, l), we need sum of w_i * w_j where s_i + s_j < 2*s_l - s_k
    # threshold[k, l] = 2*s_l - s_k
    threshold = 2.0 * s[None, :] - s[:, None]  # shape (n, n), threshold[k,l]
    
    # For each (k,l), count pairs (i,j) with pair_sums[i,j] < threshold[k,l]
    # This is O(n^4) naively, but we can vectorize
    # pair_sums shape (n,n), threshold shape (n,n)
    # We need: for each (k,l), sum_{i,j} w_i*w_j * 1[pair_sums[i,j] < threshold[k,l]]
    
    # Vectorized: expand to (n,n,n,n)
    # pair_sums[i,j] vs threshold[k,l]
    ps = pair_sums[:, :, None, None]  # (n, n, 1, 1) -> (n, n, n, n)
    th = threshold[None, None, :, :]  # (1, 1, n, n) -> (n, n, n, n)
    
    mask = ps < th  # (n, n, n, n): True if pair_sums[i,j] < threshold[k,l]
    
    ww = (w[:, None] * w[None, :])  # (n, n) = w_i * w_j
    
    # For each (k,l): sum over i,j of ww[i,j] * mask[i,j,k,l]
    # inner = ww @ mask over (i,j) for each (k,l)
    # Reshape: mask is (n_i, n_j, n_k, n_l)
    # We want result[k,l] = sum_{i,j} ww[i,j] * mask[i,j,k,l]
    
    # Use einsum or matrix multiply
    # mask reshaped: (n*n, n*n) where first index is (i,j), second is (k,l)
    mask_2d = mask.reshape(n * n, n * n)  # (i,j) x (k,l)
    ww_flat = ww.reshape(n * n)  # (i,j)
    
    # result[k,l] = sum_{i,j} ww_flat[i,j] * mask_2d[i,j, k,l]
    # = ww_flat @ mask_2d^T ... no
    # result = mask_2d.T @ ww_flat  -> shape (n*n,)
    # result_2d = result.reshape(n, n)
    
    result_2d = (mask_2d.T @ ww_flat).reshape(n, n)  # result[k,l]
    
    # P = sum_{k,l} w_k * w_l * result_2d[k,l]
    P = np.sum(w[:, None] * w[None, :] * result_2d)
    
    return P


def _neg_P(params, n):
    """Negative P for minimization. params = [s_0,...,s_{n-1}, w_0,...,w_{n-1}]"""
    s = params[:n]
    w = params[n:]
    
    # Ensure support >= 0 and weights > 0
    s = np.maximum(s, 1e-10)
    w = np.maximum(w, 1e-10)
    
    return -_compute_P(s, w)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit
    
    # Start with baseline
    n_baseline = 32
    support_baseline = np.arange(n_baseline, dtype=np.float64)
    weights_baseline = np.ones(n_baseline, dtype=np.float64) / n_baseline
    best_P = _compute_P(support_baseline, weights_baseline)
    best_support = support_baseline.copy()
    best_weights = weights_baseline.copy()
    
    # Try optimization with different n values
    for n in [4, 6, 8, 10, 12]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Initialize: spread support points, random weights
            np.random.seed(42 + restart * 1000 + n)
            
            # Support: logarithmic spacing to cover wide range
            s_init = np.exp(np.linspace(0, np.log(100), n))
            w_init = np.ones(n) / n
            
            params_init = np.concatenate([s_init, w_init])
            
            # Optimize
            try:
                result = minimize(
                    _neg_P, params_init, args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result.fun < -best_P:
                    s_opt = np.maximum(result.x[:n], 0)
                    w_opt = np.maximum(result.x[n:], 0)
                    P_opt = -result.fun
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
                        
            except Exception:
                pass
    
    # Final validation
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        # Fallback to baseline
        return support_baseline, weights_baseline
    
    if best_weights.sum() <= 1e-9:
        return support_baseline, weights_baseline
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
