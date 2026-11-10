# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _shannon_entropy(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = P.reshape(10, 10)
    # Ensure valid
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-9:
        return 0.0
    P = P / s
    
    # Project along slope r: value = x + r*y
    # P[y, x] means row=y, col=x
    ys = np.arange(10)
    xs = np.arange(10)
    Y, X = np.meshgrid(ys, xs, indexing='ij')  # Y[y,x]=y, X[y,x]=x
    
    # D = H_{-1}: distribution of x - y
    val_minus1 = X.astype(float) - Y.astype(float)
    # Pool by value
    unique_vals = np.unique(val_minus1)
    probs_minus1 = np.array([P[val_minus1 == v].sum() for v in unique_vals])
    H_minus1 = _shannon_entropy(probs_minus1)
    
    # M = max(H_0, H_1, H_2)
    # H_0: distribution of x
    val_0 = X.astype(float)
    unique_vals_0 = np.unique(val_0)
    probs_0 = np.array([P[val_0 == v].sum() for v in unique_vals_0])
    H_0 = _shannon_entropy(probs_0)
    
    # H_1: distribution of x + y
    val_1 = X.astype(float) + Y.astype(float)
    unique_vals_1 = np.unique(val_1)
    probs_1 = np.array([P[val_1 == v].sum() for v in unique_vals_1])
    H_1 = _shannon_entropy(probs_1)
    
    # H_2: distribution of x + 2y
    val_2 = X.astype(float) + 2.0 * Y.astype(float)
    unique_vals_2 = np.unique(val_2)
    probs_2 = np.array([P[val_2 == v].sum() for v in unique_vals_2])
    H_2 = _shannon_entropy(probs_2)
    
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return H_minus1 / M

def _objective(params):
    """Objective to minimize: negative of D/M ratio."""
    # params are log-space weights to ensure positivity
    P = np.exp(params)
    P = P / P.sum()
    ratio = _compute_ratio(P)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_ratio = _compute_ratio(P_uniform)
    
    # Try optimization using log-space parameterization
    # P = exp(params) / sum(exp(params))
    # Start from uniform: params = 0 for all
    x0 = np.zeros(n * n)
    
    # Use multiple starting points for robustness
    start_points = [x0]
    
    # Add some structured starting points
    # Diagonal-heavy
    for d in range(-n+1, n):
        P_struct = np.zeros((n, n))
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                P_struct[y, x] = 1.0
        if P_struct.sum() > 0:
            P_struct = P_struct / P_struct.sum()
            params_struct = np.log(P_struct.flatten() + 1e-15)
            start_points.append(params_struct)
    
    # Anti-diagonal heavy
    for d in range(-n+1, n):
        P_struct = np.zeros((n, n))
        for y in range(n):
            x = n - 1 - y + d
            if 0 <= x < n:
                P_struct[y, x] = 1.0
        if P_struct.sum() > 0:
            P_struct = P_struct / P_struct.sum()
            params_struct = np.log(P_struct.flatten() + 1e-15)
            start_points.append(params_struct)
    
    # Random starting points
    np.random.seed(42)
    for _ in range(10):
        P_rand = np.random.dirichlet(np.ones(n * n))
        params_rand = np.log(P_rand + 1e-15)
        start_points.append(params_rand)
    
    for i, x_start in enumerate(start_points):
        if time.time() > deadline - 30.0:  # Safety margin
            break
        
        try:
            result = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_ratio:
                P_opt = np.exp(result.x)
                P_opt = P_opt / P_opt.sum()
                ratio = _compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_opt.reshape(n, n)
        except Exception:
            continue
    
    # Try Nelder-Mead on the best result for further refinement
    if time.time() < deadline - 30.0:
        try:
            P_best_flat = best_P.flatten()
            x_best = np.log(P_best_flat + 1e-15)
            
            result2 = minimize(
                _objective,
                x_best,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result2.fun < -best_ratio:
                P_refined = np.exp(result2.x)
                P_refined = P_refined / P_refined.sum()
                ratio = _compute_ratio(P_refined)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_refined.reshape(n, n)
        except Exception:
            pass
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / s
    
    return best_P

# EVOLVE-BLOCK-END
