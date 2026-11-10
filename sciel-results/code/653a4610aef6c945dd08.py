# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    
    # Pair sums and probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Thresholds: thresholds[k,l] = 2*s_l - s_k
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # Sort pair sums for CDF lookup
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k,l), find index of first pair sum >= threshold
    idx = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative probability
    inner = np.zeros_like(thresholds)
    mask = idx > 0
    inner[mask] = cum_probs[idx[mask] - 1]
    
    # P = sum_{k,l} w_k * w_l * inner[k,l]
    wl = w_sorted[:, None] * w_sorted[None, :]
    P = np.sum(wl * inner)
    
    return P

def _neg_P_from_params(params, n_points):
    """Negative P for optimization. params: [log_weights (n), support (n)]."""
    log_w = params[:n_points]
    support = params[n_points:]
    
    # Convert log weights to weights
    w = np.exp(log_w - np.max(log_w))
    support = np.maximum(support, 0.0)
    
    return -_compute_P(support, w)

def _optimize_with_de(n_points, deadline, popsize=20, maxiter=100):
    """Use differential evolution to optimize support and weights."""
    # Bounds: log_weights in [-10, 0], support in [0, 100]
    bounds = [(-10, 0)] * n_points + [(0, 100)] * n_points
    
    # Time-limited differential evolution
    start_time = time.time()
    
    def _time_check():
        return time.time() > deadline
    
    try:
        result = differential_evolution(
            _neg_P_from_params,
            bounds,
            args=(n_points,),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            init='sobol',
            polish=True
        )
    except Exception:
        return None, None, 0.0
    
    if result is None or not result.success:
        return None, None, 0.0
    
    best_params = result.x
    best_P = -result.fun
    
    log_w = best_params[:n_points]
    support = best_params[n_points:]
    w = np.exp(log_w - np.max(log_w))
    support = np.maximum(support, 0.0)
    
    return support, w, best_P

def _local_refine(support, weights, deadline):
    """Local refinement using scipy minimize."""
    n = len(support)
    
    def _encode(s, w):
        log_w = np.log(w + 1e-15)
        return np.concatenate([log_w, s])
    
    def _decode(params):
        log_w = params[:n]
        s = params[n:]
        w = np.exp(log_w - np.max(log_w))
        s = np.maximum(s, 0.0)
        return s, w
    
    def _score(params):
        s, w = _decode(params)
        return -_compute_P(s, w)
    
    x0 = _encode(support, weights)
    
    try:
        result = minimize(_score, x0, method='L-BFGS-B', 
                         bounds=[(-10, 0)] * n + [(0, 100)] * n,
                         options={'maxiter': 200, 'ftol': 1e-8})
    except Exception:
        return support, weights, _compute_P(support, weights)
    
    best_params = result.x
    s, w = _decode(best_params)
    P = _compute_P(s, w)
    
    return s, w, P

def _random_restart(n_points, deadline, n_restarts=5):
    """Multiple random restarts with DE + local refinement."""
    best_support = None
    best_weights = None
    best_P = -1.0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initial point
        log_w = np.random.uniform(-8, 0, size=n_points)
        support = np.random.uniform(0, 50, size=n_points)
        w = np.exp(log_w - np.max(log_w))
        
        # DE optimization
        de_support, de_weights, de_P = _optimize_with_de(n_points, deadline, popsize=15, maxiter=50)
        
        if de_P > best_P and de_support is not None:
            best_P = de_P
            best_support = de_support.copy()
            best_weights = de_weights.copy()
        
        # Local refinement
        if best_support is not None:
            ref_support, ref_weights, ref_P = _local_refine(best_support, best_weights, deadline)
            if ref_P > best_P:
                best_P = ref_P
                best_support = ref_support.copy()
                best_weights = ref_weights.copy()
    
    return best_support, best_weights, best_P

def _structured_initial(n_points):
    """Generate structured initial configurations."""
    configs = []
    
    # Linear spacing
    s1 = np.linspace(0, 20, n_points)
    w1 = np.ones(n_points) / n_points
    configs.append((s1, w1))
    
    # Exponential spacing
    s2 = 10 * (2 ** np.arange(n_points) - 1)
    w2 = np.exp(-0.5 * np.arange(n_points))
    w2 = w2 / w2.sum()
    configs.append((s2, w2))
    
    # Bimodal: mass at low and high values
    s3 = np.concatenate([np.linspace(0, 2, n_points // 2), np.linspace(10, 30, n_points - n_points // 2)])
    w3 = np.concatenate([np.ones(n_points // 2) * 0.6 / (n_points // 2), 
                         np.ones(n_points - n_points // 2) * 0.4 / (n_points - n_points // 2)])
    configs.append((s3, w3))
    
    # Skewed: more mass at low values
    s4 = np.arange(n_points, dtype=np.float64) * 2
    w4 = np.exp(-0.3 * np.arange(n_points))
    w4 = w4 / w4.sum()
    configs.append((s4, w4))
    
    # Power law spacing
    s5 = 20 * (np.arange(n_points) / (n_points - 1)) ** 2
    w5 = np.ones(n_points) / n_points
    configs.append((s5, w5))
    
    return configs

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 610  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Try different numbers of support points
    n_configs = [4, 6, 8, 10, 12]
    
    # Phase 1: Evaluate structured initial configs
    for n in n_configs:
        if time.time() > deadline - 120:
            break
        
        configs = _structured_initial(n)
        for support, weights in configs:
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Phase 2: DE optimization for each n
    for n in n_configs:
        if time.time() > deadline - 60:
            break
        
        # Use best config for this n as starting point
        configs = _structured_initial(n)
        best_n_support = None
        best_n_weights = None
        best_n_P = -1.0
        
        for support, weights in configs:
            P = _compute_P(support, weights)
            if P > best_n_P:
                best_n_P = P
                best_n_support = support.copy()
                best_n_weights = weights.copy()
        
        # DE optimization
        de_support, de_weights, de_P = _optimize_with_de(n, deadline, popsize=20, maxiter=80)
        if de_P > best_n_P and de_support is not None:
            best_n_P = de_P
            best_n_support = de_support.copy()
            best_n_weights = de_weights.copy()
        
        # Local refinement
        if best_n_support is not None:
            ref_support, ref_weights, ref_P = _local_refine(best_n_support, best_n_weights, deadline)
            if ref_P > best_n_P:
                best_n_P = ref_P
                best_n_support = ref_support.copy()
                best_n_weights = ref_weights.copy()
        
        if best_n_P > best_P:
            best_P = best_n_P
            best_support = best_n_support.copy()
            best_weights = best_n_weights.copy()
    
    # Phase 3: Random restarts with best n
    if time.time() < deadline - 30 and best_support is not None:
        n_best = len(best_support)
        rs_support, rs_weights, rs_P = _random_restart(n_best, deadline, n_restarts=3)
        if rs_P > best_P and rs_support is not None:
            best_P = rs_P
            best_support = rs_support.copy()
            best_weights = rs_weights.copy()
    
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
