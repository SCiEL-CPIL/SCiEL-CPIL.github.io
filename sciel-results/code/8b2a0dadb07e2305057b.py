# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_p_fast(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] efficiently using vectorized operations."""
    n = len(support)
    w = weights / weights.sum()
    
    # Sort support and weights
    order = np.argsort(support)
    s_sorted = support[order]
    w_sorted = w[order]
    
    # Precompute cumulative weights for efficient prefix sums
    cum_w = np.cumsum(w_sorted)
    cum_w = np.concatenate([[0.0], cum_w])
    
    p_total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2 * s_sorted[l] - s_sorted[k]
            if threshold <= 0:
                continue
            
            # Vectorized: for each i, find max j such that s_j < threshold - s_i
            # s_i + s_j < threshold => s_j < threshold - s_i
            max_js = np.searchsorted(s_sorted, threshold - s_sorted, side='left')
            # inner = sum_i w_i * cum_w[max_j_i]
            inner = np.sum(w_sorted * cum_w[max_js])
            
            p_total += w_sorted[k] * w_sorted[l] * inner
    
    return p_total


def _neg_p(params, n_points):
    """
    Objective function to minimize (negative probability).
    params: [log_scale, alpha, beta, log_w_0, ..., log_w_{n-1}]
    Support: s_i = exp(log_scale + i * alpha) * (1 + beta * i/(n-1))
    This creates a heavy-tailed distribution with tunable tail index.
    """
    log_scale = params[0]
    alpha = params[1]
    beta = params[2]
    log_ws = params[3:]
    
    i = np.arange(n_points)
    # Support: geometric with linear perturbation for tail control
    support = np.exp(log_scale + i * alpha) * (1.0 + beta * i / max(n_points - 1, 1))
    
    weights = np.exp(log_ws)
    
    # Check for validity
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        return 1e10
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1e10
    if np.sum(weights) <= 1e-9:
        return 1e10
    
    p = _compute_p_fast(support, weights)
    return -p


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1+X2+X3 < 2*X4].
    
    Approach: Use a heavy-tailed support with optimized weights.
    The support is designed to have a long tail so that 2*X4 can often exceed
    the sum of X1+X2+X3.
    """
    n_points = 10
    
    # Multiple initial guesses for robustness
    best_result = None
    best_score = -np.inf
    
    # Initial guess 1: Heavy tail with large spread
    log_scale_init1 = np.log(0.05)
    alpha_init1 = 0.8
    beta_init1 = 0.5
    # Weights: decay to emphasize tail
    log_ws_init1 = np.linspace(0.0, -2.0, n_points)
    x0_1 = np.concatenate([[log_scale_init1], [alpha_init1], [beta_init1], log_ws_init1])
    
    # Initial guess 2: Moderate tail
    log_scale_init2 = np.log(0.1)
    alpha_init2 = 0.6
    beta_init2 = 0.3
    log_ws_init2 = np.linspace(0.0, -1.5, n_points)
    x0_2 = np.concatenate([[log_scale_init2], [alpha_init2], [beta_init2], log_ws_init2])
    
    # Initial guess 3: Very heavy tail
    log_scale_init3 = np.log(0.01)
    alpha_init3 = 1.0
    beta_init3 = 0.8
    log_ws_init3 = np.linspace(0.0, -2.5, n_points)
    x0_3 = np.concatenate([[log_scale_init3], [alpha_init3], [beta_init3], log_ws_init3])
    
    initial_guesses = [x0_1, x0_2, x0_3]
    
    for x0 in initial_guesses:
        try:
            result = minimize(
                _neg_p,
                x0,
                args=(n_points,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            
            score = -result.fun
            if score > best_score and np.isfinite(result.fun):
                best_score = score
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to a simple valid measure
        support = np.array([0.0, 1.0, 2.0, 10.0])
        weights = np.array([0.5, 0.3, 0.1, 0.1])
        return support, weights
    
    x_opt = best_result.x
    log_scale = x_opt[0]
    alpha = x_opt[1]
    beta = x_opt[2]
    log_ws = x_opt[3:]
    
    i = np.arange(n_points)
    support = np.exp(log_scale + i * alpha) * (1.0 + beta * i / max(n_points - 1, 1))
    weights = np.exp(log_ws)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Ensure valid output
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        support = np.array([0.0, 1.0, 2.0, 10.0])
        weights = np.array([0.5, 0.3, 0.1, 0.1])
    
    return support, weights


# EVOLVE-BLOCK-END
