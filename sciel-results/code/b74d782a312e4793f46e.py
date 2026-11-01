import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate test points on the sphere
    # Use a Fibonacci spiral for the test points as well, or a grid.
    # A dense Fibonacci spiral is isotropic.
    M = 2000  # Number of test points
    i_test = np.arange(M) + 0.5
    z_test = 1.0 - 2.0 * i_test / M
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    Y = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # 2. Initial configuration: Fibonacci spiral
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    X = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    def objective(params):
        # params is a flat array of 3*N
        X_curr = params.reshape(N, 3)
        
        # Normalize
        norms = np.linalg.norm(X_curr, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X_curr = X_curr / norms
        
        # Check for coincident points
        # If any two points are too close, return a large penalty
        # Calculate pairwise distances
        # dist^2 = 2 - 2 * dot
        dots = X_curr @ X_curr.T
        np.fill_diagonal(dots, 0)
        min_dot = np.min(dots)
        # If min_dot > 0.999, points are very close (angle < ~2.5 deg)
        if min_dot > 0.9999:
            return 100.0 + (0.9999 - min_dot) * 1000 # Penalty
        
        # Calculate distances from Y to nearest X
        # D is M x N matrix of cosines
        D_cos = Y @ X_curr.T
        # Clip to [-1, 1] for arccos stability
        D_cos = np.clip(D_cos, -1.0, 1.0)
        D_ang = np.arccos(D_cos)
        
        # Min over X for each Y
        D_min = np.min(D_ang, axis=1)
        
        # Soft max approximation of max(D_min)
        # beta is controlled by the outer loop
        beta = current_beta[0]
        
        # Stable log-sum-exp
        m = np.max(D_min)
        if m == 0:
            return 0.0
        log_sum_exp = m + np.log(np.sum(np.exp(beta * (D_min - m))))
        val = (1.0 / beta) * log_sum_exp
        
        return val

    current_beta = [10.0]
    
    best_X = X.copy()
    best_val = objective(X.flatten())
    
    # Annealing schedule for beta
    # Start with low beta (smooth, easy to optimize) and increase (sharp, precise)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta_val in betas:
        if time.time() > deadline:
            break
            
        current_beta[0] = beta_val
        
        # Use Nelder-Mead
        # Initial simplex is the current best X
        res = minimize(
            objective, 
            best_X.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.success or res.fun < best_val:
            # Normalize result
            X_res = res.x.reshape(N, 3)
            norms = np.linalg.norm(X_res, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_res = X_res / norms
            
            # Check validity
            if np.all(np.isfinite(X_res)):
                best_X = X_res
                best_val = res.fun
                
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_X = best_X / norms
    
    # Final validity check
    if not np.all(np.isfinite(best_X)):
        return X # Fallback to initial
    
    return best_X
