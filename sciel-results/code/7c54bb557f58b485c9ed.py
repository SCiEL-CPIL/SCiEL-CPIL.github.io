# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from collections import defaultdict


def _entropy_from_pooled(values, weights):
    """Compute Shannon entropy in bits from pooled values and weights."""
    d = defaultdict(float)
    for v, w in zip(values, weights):
        if w > 1e-15:
            d[v] += w
    total = sum(d.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in d.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)


def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    D = _entropy_from_pooled(d_vals, w)
    
    # M = max(H_0, H_1, H_2)
    h0_vals = x_flat
    h1_vals = x_flat + y_flat
    h2_vals = x_flat + 2 * y_flat
    
    H0 = _entropy_from_pooled(h0_vals, w)
    H1 = _entropy_from_pooled(h1_vals, w)
    H2 = _entropy_from_pooled(h2_vals, w)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    
    return D / M


def _negative_ratio(params):
    """Negative ratio for minimization. params are log-weights."""
    n = 10
    log_w = params.reshape(n, n)
    # Softmax to get probabilities
    log_w_max = np.max(log_w)
    exp_w = np.exp(log_w - log_w_max)
    P = exp_w / exp_w.sum()
    return -_compute_ratio(P)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Start with uniform distribution
    P_init = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_init.copy()
    best_ratio = _compute_ratio(best_P)
    
    # Parameterize as log-weights
    log_w_init = np.log(P_init.flatten())
    
    # Try multiple random initializations and optimize
    rng = np.random.default_rng(42)
    
    # First, try the uniform start
    result = minimize(
        _negative_ratio,
        log_w_init,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < -best_ratio:
        log_w_opt = result.x.reshape(n, n)
        exp_w = np.exp(log_w_opt - np.max(log_w_opt))
        P_opt = exp_w / exp_w.sum()
        ratio_opt = _compute_ratio(P_opt)
        if ratio_opt > best_ratio:
            best_P = P_opt
            best_ratio = ratio_opt
    
    # Try random initializations
    n_attempts = 20
    for i in range(n_attempts):
        if time.time() > deadline - 30:
            break
        
        # Random log-weights
        log_w_rand = rng.normal(0, 1, size=(n, n))
        
        try:
            result = minimize(
                _negative_ratio,
                log_w_rand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < -best_ratio:
                log_w_opt = result.x.reshape(n, n)
                exp_w = np.exp(log_w_opt - np.max(log_w_opt))
                P_opt = exp_w / exp_w.sum()
                ratio_opt = _compute_ratio(P_opt)
                if ratio_opt > best_ratio:
                    best_P = P_opt
                    best_ratio = ratio_opt
        except Exception:
            continue
    
    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    
    return best_P


# EVOLVE-BLOCK-END
