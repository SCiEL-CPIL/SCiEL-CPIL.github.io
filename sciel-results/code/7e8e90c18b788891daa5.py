# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from itertools import product

# Set a global deadline to avoid exceeding the evaluator's wall-clock limit
# The evaluator kills after 630s. We use 590s as a safe deadline.
GLOBAL_DEADLINE = 590.0
START_TIME = time.time()


def _time_left():
    return GLOBAL_DEADLINE - (time.time() - START_TIME)


def _compute_entropies(P):
    """
    Compute the entropies for slopes -1, 0, 1, 2.
    P is a 10x10 array where P[y, x] is the probability of (x, y).
    Note: x is the column index (0..9), y is the row index (0..9).
    
    Slope r: value = x + r*y.
    r = -1: x - y
    r = 0: x
    r = 1: x + y
    r = 2: x + 2y
    """
    n = 10
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    
    # Create meshgrids for x and y
    # P[y, x] implies y is the first index (row), x is the second index (col)
    # So X[i, j] = j, Y[i, j] = i
    X, Y = np.meshgrid(x_vals, y_vals)
    
    slopes = [-1, 0, 1, 2]
    entropies = {}
    
    for r in slopes:
        # Compute values x + r*y
        vals = X + r * Y
        # Flatten P and vals
        P_flat = P.flatten()
        vals_flat = vals.flatten()
        
        # Get unique values and their counts (sum of probabilities)
        unique_vals, inverse_indices = np.unique(vals_flat, return_inverse=True)
        
        # Sum probabilities for each unique value
        probs = np.zeros(len(unique_vals))
        np.add.at(probs, inverse_indices, P_flat)
        
        # Filter out zero probabilities
        probs = probs[probs > 1e-15]
        
        if len(probs) == 0:
            H = 0.0
        else:
            # Normalize to ensure sum is 1 (should be close to 1 already)
            probs = probs / probs.sum()
            H = -np.sum(probs * np.log2(probs))
        
        entropies[r] = H
    
    return entropies


def _objective(params):
    """
    Objective function to maximize.
    We maximize D / M, where D = H_{-1}, M = max(H_0, H_1, H_2).
    To use scipy minimize, we minimize -D/M.
    """
    if _time_left() < 5:
        return 1e10  # Penalize if time is running out
    
    n = 10
    size = n * n
    # params is a flat array of size 100
    P = params.reshape((n, n))
    
    # Ensure P is non-negative and sums to 1
    # Use a softplus-like transformation or just clip and normalize?
    # The params are unconstrained. Let's use a softmax-like approach or just simple normalization.
    # Since we are using L-BFGS-B, we can use bounds [0, 1] and then normalize?
    # No, L-BFGS-B works with bounds. Let's use bounds [0, 1] for each entry.
    # But the sum must be 1. This is a simplex constraint.
    # Standard trick: optimize over 99 parameters (last one determined by sum=1) or use penalty.
    # Better: use log-sum-exp or just simple clipping.
    # Let's try: P = np.exp(params) / sum(np.exp(params)) ? No, that's softmax.
    # Let's stick to the previous successful strategy: L-BFGS-B with annealed soft-max smoothing?
    # Or just simple: P = np.maximum(params, 0); P = P / P.sum()
    # This is non-smooth at 0.
    
    # Let's use the parameterization: P_ij = exp(theta_ij) / Z
    # This ensures P > 0 and sum P = 1.
    # Gradient is well-defined.
    
    P_raw = np.exp(params)
    P = P_raw / P_raw.sum()
    
    ent = _compute_entropies(P)
    
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    
    if M < 1e-9:
        return 1e10
        
    ratio = D / M
    return -ratio


def _objective_grad(params):
    """
    Compute gradient of the objective function.
    This is complex. Let's use finite differences for simplicity first,
    or derive it.
    Given the time constraint, finite differences might be slow but reliable.
    However, L-BFGS-B with finite differences on 100 variables is expensive.
    Let's try to derive the gradient or use a simpler method like Nelder-Mead if gradients are hard.
    But the inspiration says L-BFGS-B with gradient info is better.
    
    Let's approximate the gradient using central differences for now,
    but limit the number of restarts/steps.
    """
    if _time_left() < 5:
        return np.ones_like(params)
    
    eps = 1e-7
    grad = np.zeros_like(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
        if _time_left() < 10:
            break
            
    return grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    size = n * n
    
    # Best result found so far
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = -1e10
    
    def _update_best(P, score):
        nonlocal best_P, best_score
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Evaluate baseline
    ent = _compute_entropies(best_P)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if M > 1e-9:
        base_ratio = D / M
        _update_best(best_P, base_ratio)
    
    # Try multiple random initializations
    # Parameters are log-probabilities (theta), where P = softmax(theta)
    # Actually, P = exp(theta) / sum(exp(theta)) is softmax.
    # Uniform distribution corresponds to theta = 0 (all equal).
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    max_iter = 200
    
    for restart in range(num_restarts):
        if _time_left() < 30:
            break
            
        # Initialize theta
        # Small random noise around 0
        theta0 = rng.normal(0, 0.5, size)
        
        try:
            res = opt.minimize(
                _objective,
                theta0,
                method='L-BFGS-B',
                jac=_objective_grad,
                options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            # Extract best P from result
            theta_opt = res.x
            P_opt = np.exp(theta_opt)
            P_opt = P_opt / P_opt.sum()
            
            # Ensure validity
            if P_opt.shape != (n, n):
                continue
            if np.any(np.isnan(P_opt)) or np.any(np.isinf(P_opt)):
                continue
            if np.any(P_opt < -1e-9):
                continue
            if P_opt.sum() <= 1e-9:
                continue
                
            # Clip and normalize
            P_opt = np.clip(P_opt, 0, None)
            P_opt = P_opt / P_opt.sum()
            
            # Evaluate score
            ent = _compute_entropies(P_opt)
            D = ent[-1]
            M = max(ent[0], ent[1], ent[2])
            
            if M < 1e-9:
                continue
                
            score = D / M
            _update_best(P_opt, score)
            
        except Exception as e:
            pass
    
    # Final check and return
    P_final = best_P
    P_final = np.clip(P_final, 0, None)
    P_final = P_final / P_final.sum()
    
    return P_final


# EVOLVE-BLOCK-END
