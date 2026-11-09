# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # safety margin before 600s kill

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pairwise sums."""
    n = len(support)
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    
    # Build pairwise sums and weights
    S = s[:, None] + s[None, :]  # (n, n)
    W = w[:, None] * w[None, :]  # (n, n)
    
    # Flatten and sort
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat)
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.cumsum(W_sorted)
    W_total = W_cumsum[-1]
    
    # For each (k, l): threshold T = 2*s_l - s_k
    # inner = sum of W[i,j] where S[i,j] < T
    # Use searchsorted to find count of S_sorted < T
    T = 2.0 * s[None, :] - s[:, None]  # T[k,l] = 2*s_l - s_k
    
    # For each (k,l), find index of first S_sorted >= T
    # inner[k,l] = W_cumsum[idx-1] if idx > 0 else 0
    idx = np.searchsorted(S_sorted, T, side='left')  # (n, n)
    # inner[k,l] = W_cumsum[idx-1] where idx>0, else 0
    inner = np.zeros_like(T)
    mask = idx > 0
    inner[mask] = W_cumsum[idx[mask] - 1]
    
    # P = sum_{k,l} w[k]*w[l] * inner[k,l]
    P = np.sum(w[:, None] * w[None, :] * inner)
    return float(P)

def _objective(x, n):
    """Objective to minimize: -P. x = [support..., weights...]"""
    s = np.maximum(x[:n], 0.0)
    w = np.maximum(x[n:], 1e-12)
    if w.sum() < 1e-9:
        return 0.0
    return -_compute_P(s, w)

def _optimize_n(n, deadline):
    """Optimize for given n using differential evolution."""
    # Parameterize: support points (sorted, non-negative), weights (positive)
    # Use log-weights to ensure positivity: w_i = exp(x_{n+i}) / sum(exp(x_{n+j}))
    # Support: s_i = |x_i| to ensure non-negativity, then sort
    
    def _neg_P(params):
        s_raw = np.abs(params[:n])
        w_log = params[n:]
        # Softmax for weights
        w_max = np.max(w_log)
        w_exp = np.exp(w_log - w_max)
        w = w_exp / w_exp.sum()
        # Sort support and corresponding weights
        order = np.argsort(s_raw)
        s = s_raw[order]
        w = w[order]
        return -_compute_P(s, w)
    
    # Bounds: support in [0, 50], log-weights in [-10, 10]
    bounds = [(0, 50)] * n + [(-10, 10)] * n
    
    # Try multiple restarts with different initial populations
    best_P = -np.inf
    best_params = None
    
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        try:
            result = differential_evolution(
                _neg_P, bounds,
                maxiter=300, popsize=15,
                tol=1e-8, seed=r * 12345 + 42,
                polish=True,
                init='sobol'
            )
            if result.fun < 0:  # P > 0
                P_val = -result.fun
                if P_val > best_P:
                    best_P = P_val
                    best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        s_raw = np.abs(best_params[:n])
        w_log = best_params[n:]
        w_max = np.max(w_log)
        w_exp = np.exp(w_log - w_max)
        w = w_exp / w_exp.sum()
        order = np.argsort(s_raw)
        s = s_raw[order]
        w = w[order]
        P_val = _compute_P(s, w)
        return s, w, P_val
    return None, None, -np.inf

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = np.arange(16, dtype=np.float64)
    best_weights = np.ones(16, dtype=np.float64) / 16
    best_P = _compute_P(best_support, best_weights)
    
    # Try various support sizes
    for n in [3, 4, 5, 6, 7, 8, 10, 12, 16]:
        if time.time() > _DEADLINE - 45:
            break
        s_opt, w_opt, P_val = _optimize_n(n, _DEADLINE)
        if s_opt is not None and P_val > best_P:
            best_P = P_val
            best_support = s_opt.copy()
            best_weights = w_opt.copy()
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    if best_weights.sum() < 1e-9:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
