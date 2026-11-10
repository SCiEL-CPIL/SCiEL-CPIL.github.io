import numpy as np
import time
import sys
from scipy.optimize import minimize

def _precompute_tables():
    """Precompute index mappings for fast bincount-based ratio evaluation."""
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # x - y range: -9 to 9 (19 values)
    diff = x_idx - y_idx
    diff_idx = (diff + 9).ravel()
    
    # x range: 0 to 9 (10 values)
    x_idx_flat = x_idx.ravel()
    
    # x + y range: 0 to 18 (19 values)
    sum_idx = (x_idx + y_idx).ravel()
    
    # x + 2y range: 0 to 27 (28 values)
    sum2_idx = (x_idx + 2 * y_idx).ravel()
    
    return diff_idx, x_idx_flat, sum_idx, sum2_idx

# Pre-compute indices at module level for speed
_DIFF_IDX, _X_IDX, _SUM_IDX, _SUM2_IDX = _precompute_tables()

def _fast_ratio(P):
    """
    Compute D/M ratio efficiently using precomputed indices.
    P is a flat array of 100 elements (row-major, P[y, x]).
    Returns 0.0 if invalid.
    """
    # Check finite
    if not np.all(np.isfinite(P)):
        return 0.0
    
    # Clip negatives
    P = np.maximum(P, 0.0)
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    
    # Normalize
    P = P / total
    
    # Compute distributions
    # H_{-1}: x - y
    dist_diff = np.bincount(_DIFF_IDX, weights=P, minlength=19)
    p_diff = dist_diff[dist_diff > 0]
    H_diff = -np.sum(p_diff * np.log2(p_diff))
    
    # H_0: x
    dist_x = np.bincount(_X_IDX, weights=P, minlength=10)
    p_x = dist_x[dist_x > 0]
    H_x = -np.sum(p_x * np.log2(p_x))
    
    # H_1: x + y
    dist_sum = np.bincount(_SUM_IDX, weights=P, minlength=19)
    p_sum = dist_sum[dist_sum > 0]
    H_sum = -np.sum(p_sum * np.log2(p_sum))
    
    # H_2: x + 2y
    dist_sum2 = np.bincount(_SUM2_IDX, weights=P, minlength=28)
    p_sum2 = dist_sum2[dist_sum2 > 0]
    H_sum2 = -np.sum(p_sum2 * np.log2(p_sum2))
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _objective(x):
    """
    Objective function for minimization.
    x is a vector of length 100.
    We minimize -ratio(x).
    """
    # x is already non-negative due to bounds in L-BFGS-B
    # But we need to ensure it's on the simplex? 
    # Actually, L-BFGS-B with bounds [0, 1] doesn't enforce sum=1.
    # We need to handle the simplex constraint.
    # Strategy: Use softmax parameterization? Or just optimize over R^100 and softmax?
    # Let's use softmax parameterization to automatically satisfy P >= 0 and sum(P)=1.
    # Let x be the log-probabilities.
    # But L-BFGS-B requires bounds. Softmax is unbounded.
    # Alternative: Optimize over P directly with bounds [0, 1] and a penalty for sum != 1?
    # Or: Use a transformation.
    
    # Let's stick to optimizing P directly with bounds [0, 1].
    # The normalization happens inside _fast_ratio.
    # However, if sum(P) varies, the gradient might be weird?
    # Actually, _fast_ratio normalizes internally. So the objective is scale-invariant.
    # This means there are flat directions. L-BFGS-B might struggle.
    
    # Better approach: Parameterize with 99 free variables?
    # Or just use the unnormalized weights and let _fast_ratio handle it.
    # Since the function is homogeneous of degree 0, we can fix one variable or use a softmax.
    
    # Let's use a softmax transformation internally for the optimizer to ensure valid P.
    # The input x to _objective is the logit vector.
    # We compute P = softmax(x).
    # Then return -_fast_ratio(P).
    
    # Softmax for stability
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    P = exp_x / np.sum(exp_x)
    
    return -_fast_ratio(P)

def _jac(x):
    """
    Jacobian of the objective.
    This is tricky because of the max() and piecewise nature of entropy.
    Numerical gradient is safer and often sufficient for L-BFGS-B.
    We will not provide an analytic Jacobian to avoid bugs.
    """
    return None

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600 - 10  # 10s safety margin
    
    # Initial best: Uniform
    best_P = np.ones((n, n)) / (n * n)
    best_score = _fast_ratio(best_P.ravel())
    
    # Multi-start L-BFGS-B
    # We will run several optimizations from different starting points.
    
    # Starting points:
    # 1. Uniform (logits all 0)
    # 2. Random logits
    # 3. Structured: diagonal heavy, anti-diagonal heavy, etc.
    
    starts = []
    
    # 1. Uniform
    starts.append(np.zeros(100))
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(10):
        starts.append(np.random.randn(100) * 0.5)
        
    # 3. Structured starts
    # Diagonal: P[y, x] high when x == y
    x_idx, y_idx = np.indices((n, n))
    diag_mask = (x_idx == y_idx).ravel()
    s = np.zeros(100)
    s[diag_mask] = 1.0
    starts.append(s)
    
    # Anti-diagonal: P[y, x] high when x + y == 9
    anti_mask = (x_idx + y_idx == 9).ravel()
    s = np.zeros(100)
    s[anti_mask] = 1.0
    starts.append(s)
    
    # Corners
    s = np.zeros(100)
    s[0] = 1.0 # (0,0)
    s[99] = 1.0 # (9,9)
    s[9] = 1.0 # (0,9)
    s[90] = 1.0 # (9,0)
    starts.append(s)
    
    # Rows
    for r in range(0, 10, 2):
        s = np.zeros(100)
        s[r*10:(r+1)*10] = 1.0
        starts.append(s)
        
    # Cols
    for c in range(0, 10, 2):
        s = np.zeros(100)
        s[c::10] = 1.0
        starts.append(s)

    # Optimization loop
    for i, start_x in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective,
                start_x,
                method='L-BFGS-B',
                jac=None, # Use numerical gradient
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < -best_score:
                # Convert back to P
                x_opt = res.x
                x_max = np.max(x_opt)
                exp_x = np.exp(x_opt - x_max)
                P_candidate = exp_x / np.sum(exp_x)
                
                score = _fast_ratio(P_candidate)
                
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.reshape((n, n))
                    
        except Exception as e:
            # If optimization fails, continue to next start
            continue

    # If we have time, do a few more random restarts
    while time.time() < deadline:
        start_x = np.random.randn(100) * 1.0
        try:
            res = minimize(
                _objective,
                start_x,
                method='L-BFGS-B',
                jac=None,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < -best_score:
                x_opt = res.x
                x_max = np.max(x_opt)
                exp_x = np.exp(x_opt - x_max)
                P_candidate = exp_x / np.sum(exp_x)
                
                score = _fast_ratio(P_candidate)
                
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.reshape((n, n))
        except:
            pass

    return best_P
