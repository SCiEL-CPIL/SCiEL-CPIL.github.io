# EVOLVE-BLOCK-START
import numpy as np
import time


def _project_entropy(P, r):
    """Compute Shannon entropy (bits) of projection x + r*y over P[y,x]."""
    n = P.shape[0]
    # P[y, x] for x,y in 0..n-1
    # Projection value: x + r*y
    # For integer slopes, values are discrete
    vals = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            vals[y, x] = x + r * y
    # Pool by unique values
    unique_vals = np.unique(vals)
    probs = np.zeros(len(unique_vals))
    for i, uv in enumerate(unique_vals):
        mask = (vals == uv)
        probs[i] = P[mask].sum()
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _gradient_H_minus1(P, n):
    """Gradient of H_{-1} w.r.t. P[y,x]. H_{-1} = H(x - y)."""
    # For each cell (y,x), contribution to value v = x - y
    # dH/dP[y,x] = -log2(q_v) - 1/ln(2) where q_v = sum of P over cells with same v
    grad = np.zeros_like(P)
    vals = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            vals[y, x] = x - y
    unique_vals = np.unique(vals)
    for uv in unique_vals:
        mask = (vals == uv)
        q = P[mask].sum()
        if q > 1e-15:
            # dH/dP = -(log2(q) + 1/ln(2))
            grad[mask] = -(np.log2(q) + 1.0 / np.log(2))
    return grad


def _gradient_H_r(P, r, n):
    """Gradient of H_r w.r.t. P[y,x]. H_r = H(x + r*y)."""
    grad = np.zeros_like(P)
    vals = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            vals[y, x] = x + r * y
    unique_vals = np.unique(vals)
    for uv in unique_vals:
        mask = (vals == uv)
        q = P[mask].sum()
        if q > 1e-15:
            grad[mask] = -(np.log2(q) + 1.0 / np.log(2))
    return grad


def _project_simplex(P, n):
    """Project P onto simplex: non-negative, sum to 1."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 1e-15:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize P to maximize H_{-1} / max(H_0, H_1, H_2) using DC programming
    with alternating gradient ascent and adaptive thresholding.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 620  # safety margin
    
    # Initialize with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Compute initial ratio
    def compute_ratio(P):
        D = _project_entropy(P, -1)
        M = max(_project_entropy(P, 0), _project_entropy(P, 1), _project_entropy(P, 2))
        if M < 1e-9:
            return 0.0
        return D / M
    
    best_P = P.copy()
    best_ratio = compute_ratio(P)
    
    # DC programming: alternate between maximizing H_{-1} and minimizing max(H_0,H_1,H_2)
    # Use penalty method: maximize H_{-1} - lambda * max(H_0, H_1, H_2)
    
    slopes_denom = [0, 1, 2]
    
    # Adaptive lambda
    lambda_val = 1.0
    step_size = 0.05
    patience = 0
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Compute gradients
        grad_num = _gradient_H_minus1(P, n)
        grad_denom = np.zeros_like(P)
        for r in slopes_denom:
            grad_denom += _gradient_H_r(P, r, n)
        
        # DC objective gradient: grad_num - lambda * grad_denom
        grad = grad_num - lambda_val * grad_denom
        
        # Gradient step
        P_new = P + step_size * grad
        P_new = _project_simplex(P_new, n)
        
        # Compute new ratio
        new_ratio = compute_ratio(P_new)
        
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_P = P_new.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive step size
        if patience > 100:
            step_size *= 0.9
            patience = 0
        
        # Adaptive lambda: increase if denominator is too large relative to numerator
        D = _project_entropy(P, -1)
        M = max(_project_entropy(P, 0), _project_entropy(P, 1), _project_entropy(P, 2))
        if M > 1e-9:
            # Adjust lambda to balance
            target_ratio = 1.0  # want D/M ~ 1
            if D / M < target_ratio:
                lambda_val *= 1.05
            else:
                lambda_val *= 0.95
            lambda_val = np.clip(lambda_val, 0.1, 10.0)
        
        P = P_new
    
    return best_P


# EVOLVE-BLOCK-END
