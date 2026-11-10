import numpy as np
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 600  # seconds
    
    # Try different K values, starting from a reasonable size
    # K is the max key in the domain
    # We want to maximize the score. Larger K allows more terms in the sum.
    # However, the constraint matrix becomes larger.
    
    K = 2000
    
    # Variables: f_2, ..., f_K
    n_vars = K - 1
    
    # Objective coefficients: c_k = log(k) / k
    # We want to minimize sum(c_k * f_k) which is equivalent to maximizing -sum(c_k * f_k)
    c = np.array([np.log(k) / k for k in range(2, K + 1)])
    
    # Inequality constraints: A_ub @ f <= b_ub
    # For each x in [1, 10*K]:
    # sum_{k=2}^K f_k * (floor(x/k) - x/k) <= 1
    
    max_x = 10 * K
    n_constraints = max_x
    
    # Precomputing A_ub might be memory intensive for large K
    # n_constraints * n_vars = 10*K * K = 10 K^2
    # For K=2000, 10 * 4e6 = 40e6 floats. 40e6 * 8 bytes = 320 MB. This is fine.
    # For K=5000, 10 * 25e6 = 250e6 floats. 250e6 * 8 = 2 GB. This might be tight or slow.
    # Let's stick to K=2000 or 3000 first.
    
    # Construct A_ub
    # A_ub[i, j] corresponds to constraint i (x = i+1) and variable j (k = j+2)
    # A_ub[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2)
    
    # We can build this row by row or use broadcasting if efficient.
    # Broadcasting:
    # x_vals = np.arange(1, max_x + 1)[:, None]  # shape (max_x, 1)
    # k_vals = np.arange(2, K + 1)[None, :]      # shape (1, n_vars)
    # A_ub = np.floor(x_vals / k_vals) - x_vals / k_vals
    
    # This creates a large intermediate array.
    # Let's check memory.
    # If K=2000, max_x=20000. 20000 * 1999 ~ 40M elements. OK.
    
    x_vals = np.arange(1, max_x + 1, dtype=np.float64)[:, None]
    k_vals = np.arange(2, K + 1, dtype=np.float64)[None, :]
    
    # Calculate A_ub
    # Note: floor(x/k) is integer, x/k is float.
    A_ub = np.floor(x_vals / k_vals) - (x_vals / k_vals)
    
    b_ub = np.ones(max_x)
    
    # Bounds for f_k: [-10, 10]
    bounds = [(-10, 10)] * n_vars
    
    # Solve LP
    # method='highs' is the default and efficient in recent scipy
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        f_opt = res.x
        # f_opt contains f_2 ... f_K
        # We need to construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        
        # f_1 will be adjusted by evaluator, so we can set it to 0
        # The evaluator will compute S = sum(f_k/k) and set f_1 = f_1 - S
        # Since we don't know exactly how the evaluator handles the "clipped" values vs "original" for S,
        # but the prompt says "values are clipped... Then f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values",
        # it implies S is calculated from the clipped f_2...f_K and clipped f_1 (which is 0).
        # So S = 0/1 + sum_{k=2}^K f_k/k.
        # New f_1 = 0 - S = - sum_{k=2}^K f_k/k.
        # This matches our derivation.
        
        values[1:] = f_opt
        values[0] = 0.0  # Placeholder for f_1
        
        return domain, values
    else:
        # Fallback to baseline if LP fails
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
