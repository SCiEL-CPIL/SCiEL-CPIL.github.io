# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = 0.0
    for c in counts_d:
        if c > 0:
            H_d -= c * np.log2(c)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        counts = np.zeros(n + r * (n - 1) + 1)
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        for v in range(int(min_val), int(max_val) + 1):
            mask = (vals == v)
            counts[v - min_val] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _coordinate_descent_step(P, max_iter=200):
    """Optimize one cell at a time via coordinate descent."""
    n = 10
    P = P.copy()
    best_ratio = _compute_ratio(P)
    
    for _ in range(max_iter):
        improved = False
        # Randomize order to avoid cycling
        order = np.random.permutation(n * n)
        for idx in order:
            i, j = divmod(idx, n)
            old_val = P[i, j]
            
            # Try a few candidate values for this cell
            candidates = [0.0, old_val * 0.5, old_val * 1.5, old_val * 0.1, old_val * 2.0]
            # Also try absolute values
            candidates += [0.001, 0.01, 0.05, 0.1]
            # Filter to reasonable range
            candidates = [c for c in candidates if 0 <= c <= 0.5]
            
            best_cand = old_val
            best_cand_ratio = best_ratio
            
            for c in candidates:
                P[i, j] = c
                P_norm = _clip_and_normalize(P)
                r = _compute_ratio(P_norm)
                if r > best_cand_ratio + 1e-12:
                    best_cand_ratio = r
                    best_cand = c
            
            if best_cand != old_val:
                P[i, j] = best_cand
                P = _clip_and_normalize(P)
                best_ratio = best_cand_ratio
                improved = True
        
        if not improved:
            break
    
    return P, best_ratio


def _random_restart_init():
    """Generate a diverse random initial distribution."""
    n = 10
    method = random.choice(['uniform', 'random', 'structured', 'low_rank'])
    
    if method == 'uniform':
        P = np.ones((n, n)) / (n * n)
    elif method == 'random':
        P = np.random.dirichlet(np.ones(n * n))
        P = P.reshape(n, n)
    elif method == 'structured':
        # Emphasize anti-diagonal (x - y = 0)
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = abs(x - y)
                P[y, x] = np.exp(-d * 0.3)
        P = _clip_and_normalize(P)
    else:  # low_rank
        # Random low-rank
        A = np.random.randn(n, 2)
        B = np.random.randn(2, n)
        P = np.abs(A @ B)
        P = _clip_and_normalize(P)
    
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses random restarts with coordinate descent to find a better distribution than uniform.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 580.0  # seconds, with safety margin
    
    # Start from uniform distribution
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Random restarts with coordinate descent
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 0
    while time.time() - start_time < deadline:
        # Generate a new random initialization
        P_init = _random_restart_init()
        
        # Run coordinate descent from this initialization
        P_opt, ratio_opt = _coordinate_descent_step(P_init, max_iter=100)
        
        if ratio_opt > best_ratio + 1e-9:
            best_P = P_opt.copy()
            best_ratio = ratio_opt
        
        num_restarts += 1
        
        # Check time
        if time.time() - start_time > deadline - 30:
            break
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
