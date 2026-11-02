import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconvolution(h):
    # Linear autoconvolution: result has length 2*N-1
    return np.convolve(h, h, mode='full')

def _objective(x):
    # x: log of heights (so heights = exp(x) >= 0)
    # We want to maximize (sum h_i^2) / ((sum |h_i|) * max(autoconv))
    # Equivalently minimize its reciprocal
    h = np.exp(x)
    sum_sq = np.sum(h * h)
    sum_abs = np.sum(h)
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 1e10
    ratio = sum_sq / (sum_abs * max_conv)
    return -ratio  # minimize negative to maximize

def _objective_and_grad(x):
    h = np.exp(x)
    sum_sq = np.sum(h * h)
    sum_abs = np.sum(h)
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 1e10, np.zeros_like(x)
    
    ratio = sum_sq / (sum_abs * max_conv)
    # Gradient with respect to x (log heights)
    # d(ratio)/d(h_i) * h_i
    # Let's use finite differences for robustness
    grad = np.zeros_like(x)
    eps = 1e-6
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_objective(x_plus) - _objective(x_minus)) / (2 * eps)
    return _objective(x), grad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Initial guess: sparse Sidon-like pattern
    # A Sidon set minimizes collisions in autoconvolution
    # For N=50, a good sparse pattern with equal heights on a Sidon set
    # Max Sidon size in [0,49] is around 7-8
    # Let's try a few initial patterns and pick the best one
    
    patterns = []
    
    # Pattern 1: Equal heights on a known Sidon set
    # {0, 1, 4, 10, 18, 28, 40} - let me verify this is Sidon
    # Actually, let's just use random sparse patterns and optimize
    for _ in range(20):
        # Random sparse binary pattern with 3-8 ones
        k = np.random.randint(3, 9)
        idx = np.random.choice(N, k, replace=False)
        x0 = np.full(N, -5.0)  # Small heights
        x0[idx] = 0.0  # Height 1.0 at selected positions
        patterns.append(x0)
    
    # Also try the uniform pattern
    patterns.append(np.zeros(N))
    
    # Also try a few structured patterns
    # Pattern: peaks at positions that are well-separated
    for spacing in [3, 4, 5, 6, 7]:
        x0 = np.full(N, -5.0)
        x0[::spacing] = 0.0
        patterns.append(x0)
    
    best_x = None
    best_val = -1e10
    
    for x0 in patterns:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-6})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: uniform
        return [1.0] * N
    
    # Convert back to heights
    h = np.exp(best_x)
    
    # Ensure nonnegative (should be by construction)
    h = np.maximum(h, 0.0)
    
    # Normalize: the ratio is scale-invariant, so any scaling works
    # But let's keep heights reasonable
    if np.max(h) > 0:
        h = h / np.max(h)
    
    return list(h)
