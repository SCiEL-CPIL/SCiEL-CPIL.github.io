# EVOLVE-BLOCK-START
import numpy as np
import time


def _shannon_entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array, ignoring zeros."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    # P[y, x] where y is row index, x is col index
    # x + r*y for each slope r
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    # Map to indices: x-y + 9 gives 0..18
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = _shannon_entropy_from_counts(counts_d)
    
    # M: max of H(x), H(x+y), H(x+2y)
    # H(x): distribution of x
    counts_x = np.zeros(n)
    for x in range(n):
        counts_x[x] = P[:, x].sum()
    H_x = _shannon_entropy_from_counts(counts_x)
    
    # H(x+y): values 0..18
    vals_xy = X + Y
    counts_xy = np.zeros(19)
    for v in range(19):
        mask = (vals_xy == v)
        counts_xy[v] = P[mask].sum()
    H_xy = _shannon_entropy_from_counts(counts_xy)
    
    # H(x+2y): values 0..27
    vals_x2y = X + 2 * Y
    counts_x2y = np.zeros(28)
    for v in range(28):
        mask = (vals_x2y == v)
        counts_x2y[v] = P[mask].sum()
    H_x2y = _shannon_entropy_from_counts(counts_x2y)
    
    M = max(H_x, H_xy, H_x2y)
    if M <= 1e-9:
        return 0.0
    return H_d / M


def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.maximum(P, 0.0)
    s = P.sum()
    if s <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / s


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    n = 10
    # Start with uniform
    P_best = np.ones((n, n), dtype=np.float64) / (n * n)
    best_ratio = _compute_ratio(P_best)
    
    # Use scipy if available for optimization
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        # Parameterize: use log-space for positivity
        # p = exp(log_p) / sum(exp(log_p))
        # We optimize over log_p (100 parameters)
        
        def objective(params):
            log_p = params.reshape(10, 10)
            # Softmax for stability
            log_p_shifted = log_p - log_p.max()
            p_unnorm = np.exp(log_p_shifted)
            p = p_unnorm / p_unnorm.sum()
            return -_compute_ratio(p)
        
        # Initial guess: uniform in log space
        x0 = np.zeros(100)
        
        # Try differential evolution (robust, global)
        bounds = [(-5, 5)] * 100
        
        def time_check():
            return time.time() >= deadline
        
        # Run with time limit
        best_result = None
        try:
            result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=1000, tol=1e-6,
                polish=True, init='sobol',
                maxfev=5000
            )
            if time.time() < deadline:
                log_p_opt = result.x.reshape(10, 10)
                log_p_shifted = log_p_opt - log_p_opt.max()
                P_candidate = np.exp(log_p_shifted)
                P_candidate = P_candidate / P_candidate.sum()
                ratio_candidate = _compute_ratio(P_candidate)
                if ratio_candidate > best_ratio:
                    best_ratio = ratio_candidate
                    P_best = P_candidate
        except Exception:
            pass
        
    except ImportError:
        pass
    
    # If scipy didn't help or wasn't available, try a simple iterative refinement
    if time.time() < deadline - 30:
        # Local search: try perturbing the distribution
        P_current = P_best.copy()
        for iteration in range(10000):
            if time.time() >= deadline:
                break
            
            # Random perturbation
            P_perturbed = P_current.copy()
            # Add small random noise
            noise = np.random.normal(0, 0.05, (10, 10))
            P_perturbed = P_current + noise
            P_perturbed = _clip_and_normalize(P_perturbed)
            
            ratio_pert = _compute_ratio(P_perturbed)
            if ratio_pert > best_ratio:
                best_ratio = ratio_pert
                P_current = P_perturbed
                P_best = P_perturbed.copy()
    
    # Final cleanup
    P_final = _clip_and_normalize(P_best)
    
    # Validate
    assert P_final.shape == (10, 10)
    assert np.all(np.isfinite(P_final))
    assert np.all(P_final >= -1e-9)
    assert P_final.sum() > 1e-9
    
    return P_final


# EVOLVE-BLOCK-END
