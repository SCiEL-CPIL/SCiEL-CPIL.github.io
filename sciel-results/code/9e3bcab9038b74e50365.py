"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 150s kill
    
    # Start with flat function as baseline
    best_f = np.ones(N)
    best_value = np.inf
    
    # Compute the value for a given function
    def compute_value(f):
        # Discrete autoconvolution
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        sum_f = np.sum(f)
        if sum_f == 0:
            return np.inf
        # Normalized maximum: max(f*f) / (sum(f))^2 * N
        return max_conv / (sum_f ** 2) * N
    
    # Evaluate baseline
    best_value = compute_value(best_f)
    
    # Try smooth optimization using log-sum-exp as surrogate for max
    # We want to minimize max(conv) subject to sum(f) = 1 (normalization)
    # Use log-sum-exp as smooth approximation to max
    
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return best_value
        
        f = np.exp(x)  # Ensure positivity
        # Normalize to sum to 1
        f = f / np.sum(f)
        
        conv = np.convolve(f, f, mode='full')
        
        # Use log-sum-exp as smooth surrogate for max
        beta = 100.0
        max_val = np.logaddexp.reduce(conv) / beta
        
        return max_val
    
    def constraint_sum(x):
        f = np.exp(x)
        return np.sum(f) - 1.0
    
    # Random restarts
    n_restarts = 20
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Random initial point
        x0 = np.random.randn(N) * 0.1
        
        try:
            result = minimize(
                objective, x0,
                method='SLSQP',
                constraints=[{'type': 'eq', 'fun': constraint_sum}],
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success:
                f_opt = np.exp(result.x)
                f_opt = f_opt / np.sum(f_opt)
                val = compute_value(f_opt)
                
                if val < best_value:
                    best_value = val
                    best_f = f_opt.copy()
                    
        except Exception:
            continue
    
    # Convert to the expected format: sequence of nonnegative step heights
    # The heights should be nonnegative
    result_heights = np.maximum(best_f, 0)
    
    # Ensure it's a list of floats
    return [float(h) for h in result_heights]
# EVOLVE-BLOCK-END
