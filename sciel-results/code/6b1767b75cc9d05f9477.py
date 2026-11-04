import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Objective function: minimize -ratio
    # x: array of first N-1 gaps
    def objective(x):
        # Check for invalid gaps
        if np.any(x <= 1e-9):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        if deltas[N-1] <= 1e-9:
            return 0.0
            
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate minimum triangle area numerator
        # Vectorized approach for better speed
        # Generate indices for all triples i < j < k
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        # Extract angles
        phi_i = phi[i_flat]
        phi_j = phi[j_flat]
        phi_k = phi[k_flat]
        
        a = phi_j - phi_i
        b = phi_k - phi_j
        c = 2 * np.pi - a - b
        
        vals = np.sin(a) + np.sin(b) + np.sin(c)
        min_num = np.min(vals)
        
        # Hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 1e-9:
            return 0.0
            
        ratio = min_num / den
        return -ratio

    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_score = -1e10
    best_x = None
    
    # Multiple restarts to avoid local minima
    # 1. Uniform start
    starts = [np.full(N-1, 2 * np.pi / N)]
    
    # 2. Random perturbations of uniform
    np.random.seed(42)
    for _ in range(10):
        noise = np.random.normal(0, 0.1, N-1)
        start = np.full(N-1, 2 * np.pi / N) + noise
        # Ensure positive and sum < 2pi (approx)
        # Clip to bounds
        start = np.clip(start, 1e-5, 2*np.pi - 1e-5)
        # Adjust last gap implicitly by the optimizer, but initial x must be valid
        # We just need x to be within bounds for the optimizer to start
        starts.append(start)

    for start in starts:
        # Check if start is valid for the objective (sum < 2pi)
        if np.sum(start) >= 2 * np.pi - 1e-5:
             continue
             
        result = minimize(
            objective, 
            start, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={
                'xatol': 1e-8, 
                'fatol': 1e-8, 
                'maxiter': 5000, 
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_score:
             # result.fun is -ratio, so we want it to be "less negative" (closer to 0) or higher?
             # Wait, maximize ratio -> minimize -ratio.
             # So we want the result with the HIGHEST -ratio? No.
             # If ratio is 0.1, -ratio is -0.1.
             # If ratio is 0.2, -ratio is -0.2.
             # We want to maximize ratio, so we want to MINIMIZE -ratio.
             # So the "best" result has the smallest (most negative) value of result.fun?
             # No, if ratio is higher, -ratio is lower (more negative).
             # Example: Ratio 0.1 -> -0.1. Ratio 0.2 -> -0.2.
             # 0.2 > 0.1, but -0.2 < -0.1.
             # So we want the result with the MINIMUM value of result.fun.
             
             if result.fun < best_score:
                 best_score = result.fun
                 best_x = result.x
    
    if best_x is None:
        # Fallback
        best_x = np.full(N-1, 2 * np.pi / N)
        
    # Construct points
    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = best_x
    opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
    
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
