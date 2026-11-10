# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _shannon_entropy_bits(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y, x]."""
    n = 10
    # P[y, x] where y is row, x is col
    # x ranges 0..9 (columns), y ranges 0..9 (rows)
    
    # Compute projections
    # x - y (D)
    diff_vals = np.array([x - y for y in range(n) for x in range(n)])
    diff_probs = P.flatten()[np.argsort(diff_vals)]
    # Better: use bincount
    diff_idx = np.array([x - y for y in range(n) for x in range(n)])
    diff_counts = np.bincount(diff_idx + n, weights=P.flatten(), minlength=2*n-1)
    D = _shannon_entropy_bits(diff_counts)
    
    # x (H_0)
    x_idx = np.array([x for y in range(n) for x in range(n)])
    x_counts = np.bincount(x_idx, weights=P.flatten(), minlength=n)
    H0 = _shannon_entropy_bits(x_counts)
    
    # x + y (H_1)
    sum_idx = np.array([x + y for y in range(n) for x in range(n)])
    sum_counts = np.bincount(sum_idx, weights=P.flatten(), minlength=2*n-1)
    H1 = _shannon_entropy_bits(sum_counts)
    
    # x + 2y (H_2)
    sum2_idx = np.array([x + 2*y for y in range(n) for x in range(n)])
    sum2_counts = np.bincount(sum2_idx, weights=P.flatten(), minlength=2*n)
    H2 = _shannon_entropy_bits(sum2_counts)
    
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return -1e10  # invalid
    return D / M


def _objective(params, n=10):
    """Objective function to minimize (negative of D/M)."""
    log_w = params.reshape(n, n)
    # Softmax to get probabilities
    log_w_max = np.max(log_w)
    w = np.exp(log_w - log_w_max)
    total = np.sum(w)
    if total < 1e-15:
        return 1e10
    P = w / total
    ratio = _compute_ratio(P)
    if ratio < -1e9:
        return 1e10
    return -ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on a 10x10 grid.
    Returns a 10x10 array P of finite weights, P[y, x], normalized to sum 1.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Precompute index arrays for efficiency
    x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(n))
    diff_idx = (x_grid - y_grid).flatten() + n
    x_idx = x_grid.flatten()
    sum_idx = (x_grid + y_grid).flatten()
    sum2_idx = (x_grid + 2*y_grid).flatten()
    
    def _ratio_from_logw(log_w):
        """Compute D/M from log-weights."""
        log_w_max = np.max(log_w)
        w = np.exp(log_w - log_w_max)
        total = np.sum(w)
        if total < 1e-15:
            return -1e10
        P = w / total
        P_flat = P.flatten()
        
        diff_counts = np.bincount(diff_idx, weights=P_flat, minlength=2*n-1)
        D = _shannon_entropy_bits(diff_counts)
        
        x_counts = np.bincount(x_idx, weights=P_flat, minlength=n)
        H0 = _shannon_entropy_bits(x_counts)
        
        sum_counts = np.bincount(sum_idx, weights=P_flat, minlength=2*n-1)
        H1 = _shannon_entropy_bits(sum_counts)
        
        sum2_counts = np.bincount(sum2_idx, weights=P_flat, minlength=2*n)
        H2 = _shannon_entropy_bits(sum2_counts)
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            return -1e10
        return D / M
    
    best_ratio = -1e10
    best_log_w = np.zeros((n, n))
    
    # Try multiple initializations
    np.random.seed(42)
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros((n, n)))
    
    # 2. Random initializations
    for _ in range(5):
        initializations.append(np.random.randn(n, n) * 0.5)
    
    # 3. Structured initializations - concentrate on certain patterns
    # Pattern: x in middle, y spread
    for x_center in [3, 4, 5, 6]:
        init = np.zeros((n, n))
        init[:, x_center] = 1.0
        initializations.append(init)
    
    # Pattern: y in middle, x spread
    for y_center in [3, 4, 5, 6]:
        init = np.zeros((n, n))
        init[y_center, :] = 1.0
        initializations.append(init)
    
    # Diagonal patterns
    for offset in [-2, -1, 0, 1, 2]:
        init = np.zeros((n, n))
        for i in range(n):
            j = i + offset
            if 0 <= j < n:
                init[i, j] = 1.0
        if np.sum(init) > 0:
            initializations.append(init)
    
    # Anti-diagonal patterns
    for offset in [-2, -1, 0, 1, 2]:
        init = np.zeros((n, n))
        for i in range(n):
            j = n - 1 - i + offset
            if 0 <= j < n:
                init[i, j] = 1.0
        if np.sum(init) > 0:
            initializations.append(init)
    
    # Gaussian-like patterns
    for cx in [2, 4, 6]:
        for cy in [2, 4, 6]:
            init = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    init[i, j] = np.exp(-((i-cy)**2 + (j-cx)**2) / 4.0)
            initializations.append(init)
    
    # Sparse patterns - few cells
    for _ in range(10):
        init = np.zeros((n, n))
        num_cells = np.random.randint(3, 8)
        cells = np.random.choice(n*n, size=num_cells, replace=False)
        init.flatten()[cells] = 1.0
        initializations.append(init)
    
    # Optimize each initialization
    for init_idx, init_log_w in enumerate(initializations):
        if time.time() - start_time > deadline - 30:
            break
        
        current_log_w = init_log_w.copy()
        current_ratio = _ratio_from_logw(current_log_w)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_log_w = current_log_w.copy()
        
        # Use Nelder-Mead for local optimization (no gradient needed)
        try:
            result = minimize(
                _objective,
                current_log_w.flatten(),
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if time.time() - start_time > deadline - 30:
                break
                
            new_ratio = _ratio_from_logw(result.x.reshape(n, n))
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_log_w = result.x.reshape(n, n).copy()
                
        except Exception:
            pass
    
    # Convert best log-weights to probabilities
    log_w_max = np.max(best_log_w)
    w = np.exp(best_log_w - log_w_max)
    total = np.sum(w)
    if total < 1e-15:
        # Fallback to uniform
        P = np.ones((n, n)) / (n * n)
    else:
        P = w / total
    
    # Clip and normalize
    P = np.clip(P, 0, None)
    P_sum = np.sum(P)
    if P_sum < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / P_sum
    
    # Ensure finite
    P = np.where(np.isfinite(P), P, 0.0)
    P_sum = np.sum(P)
    if P_sum < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / P_sum
    
    return P


# EVOLVE-BLOCK-END
