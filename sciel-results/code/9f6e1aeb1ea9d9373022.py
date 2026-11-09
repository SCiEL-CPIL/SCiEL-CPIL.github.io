# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _compute_gradient(P):
    """Compute gradient of D/M with respect to P using finite differences."""
    n = 10
    eps = 1e-6
    grad = np.zeros((n, n))
    base_ratio = _compute_ratio(P)
    
    for i in range(n):
        for j in range(n):
            P_plus = P.copy()
            P_plus[i, j] += eps
            P_plus = _clip_and_normalize(P_plus)
            r_plus = _compute_ratio(P_plus)
            
            P_minus = P.copy()
            P_minus[i, j] -= eps
            P_minus = _clip_and_normalize(P_minus)
            r_minus = _compute_ratio(P_minus)
            
            grad[i, j] = (r_plus - r_minus) / (2 * eps)
    
    return grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses gradient-based optimization to find a better distribution than uniform.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin
    
    # Start from uniform distribution
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Gradient ascent with adaptive step size
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < deadline - 30:  # 30s safety margin
        # Compute gradient
        grad = _compute_gradient(P)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Try step
        P_new = P + step_size * grad
        P_new = _clip_and_normalize(P_new)
        new_ratio = _compute_ratio(P_new)
        
        if new_ratio > best_ratio + 1e-9:
            best_P = P_new.copy()
            best_ratio = new_ratio
            no_improve_count = 0
            P = P_new
        else:
            no_improve_count += 1
            # Reduce step size if no improvement
            if no_improve_count >= 5:
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-6:
                    break
        
        # Check time
        if time.time() - start_time > deadline - 60:
            break
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
