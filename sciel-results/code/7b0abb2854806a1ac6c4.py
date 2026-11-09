# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    We maximize D/M where:
    - D = H_{-1} (entropy of distribution of x - y)
    - M = max(H_0, H_1, H_2) (max of entropies of x, x+y, x+2y)
    
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    n = 10
    
    # Precompute the projection indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    # X[i, j] = j (x-coordinate), Y[i, j] = i (y-coordinate)
    # P[y, x] = P[i, j] where i=y, j=x
    
    # Projection values:
    # For slope r: value = x + r*y
    # r = -1: x - y
    # r = 0: x
    # r = 1: x + y
    # r = 2: x + 2y
    
    proj_m1 = X - Y  # x - y, values in -9..9
    proj_0 = X       # x, values in 0..9
    proj_1 = X + Y   # x + y, values in 0..18
    proj_2 = X + 2*Y # x + 2y, values in 0..27
    
    # Flatten for computation
    proj_m1_flat = proj_m1.ravel()
    proj_0_flat = proj_0.ravel()
    proj_1_flat = proj_1.ravel()
    proj_2_flat = proj_2.ravel()
    
    def entropy_from_values(values_flat, p_flat):
        """Compute Shannon entropy in bits from projection values and probabilities."""
        # Get unique values and their indices
        unique_vals, inverse = np.unique(values_flat, return_inverse=True)
        # Sum probabilities for each unique value
        p_sum = np.zeros(len(unique_vals))
        np.add.at(p_sum, inverse, p_flat)
        # Filter out zero probabilities
        mask = p_sum > 1e-15
        p_nonzero = p_sum[mask]
        if np.sum(p_nonzero) < 1e-15:
            return 0.0
        # Normalize to ensure sum is 1 (handles numerical issues)
        p_nonzero = p_nonzero / p_nonzero.sum()
        return float(-np.sum(p_nonzero * np.log2(p_nonzero)))
    
    def compute_ratio(P):
        """Compute the ratio D/M for a given distribution P."""
        p_flat = P.ravel()
        # Clip negative values and normalize
        p_flat = np.clip(p_flat, 0, None)
        total = p_flat.sum()
        if total < 1e-15:
            return 0.0
        p_flat = p_flat / total
        
        H_m1 = entropy_from_values(proj_m1_flat, p_flat)
        H_0 = entropy_from_values(proj_0_flat, p_flat)
        H_1 = entropy_from_values(proj_1_flat, p_flat)
        H_2 = entropy_from_values(proj_2_flat, p_flat)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_m1 / M
    
    def objective(params):
        """Objective function for optimization: minimize negative ratio."""
        P = np.exp(params)  # Softmax-like parametrization to ensure positivity
        P = P / P.sum()  # Normalize
        return -compute_ratio(P)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 620.0  # Leave 10s margin
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_score = compute_ratio(best_P)
    
    # Try some heuristic initializations
    # 1. Weight by |x-y| to flatten x-y distribution
    weights1 = np.abs(X - Y) + 1
    P1 = weights1 / weights1.sum()
    score1 = compute_ratio(P1)
    if score1 > best_score:
        best_score = score1
        best_P = P1.copy()
    
    # 2. Weight to concentrate on diagonal (x=y) for low H_0, H_1, H_2
    # Actually, we want high H_{-1} and low max(H_0, H_1, H_2)
    # H_{-1} is high when x-y is spread out
    # H_0 is high when x is spread out
    # H_1 is high when x+y is spread out
    # H_2 is high when x+2y is spread out
    
    # Try a distribution that spreads x-y but concentrates x, x+y, x+2y
    # This is tricky. Let's use optimization.
    
    # Use Nelder-Mead optimization
    # Parametrize P as exp(params) / sum(exp(params))
    P_init = best_P.copy()
    params_init = np.log(P_init + 1e-10)
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, params_init, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if time.time() < deadline:
            P_opt = np.exp(result.x)
            P_opt = P_opt / P_opt.sum()
            P_opt = P_opt.reshape((n, n))
            score_opt = compute_ratio(P_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
    except Exception:
        pass
    
    # Try more iterations with different starting points if time permits
    if time.time() < deadline - 60:
        # Try starting from uniform
        P_init2 = np.ones((n, n), dtype=np.float64) / (n * n)
        params_init2 = np.log(P_init2 + 1e-10)
        try:
            result2 = minimize(objective, params_init2, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            if time.time() < deadline:
                P_opt2 = np.exp(result2.x)
                P_opt2 = P_opt2 / P_opt2.sum()
                P_opt2 = P_opt2.reshape((n, n))
                score_opt2 = compute_ratio(P_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_P = P_opt2.copy()
        except Exception:
            pass
    
    # Final cleanup: clip and normalize
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END
