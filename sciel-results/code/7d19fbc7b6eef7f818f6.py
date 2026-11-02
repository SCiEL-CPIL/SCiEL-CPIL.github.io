import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 100
    # Assume the metric considers the first K shifts, likely K=5 based on "C5"
    K = 5
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    def objective(h):
        # Compute overlaps for shifts 1..K
        # O_k = sum h_i (1 - h_{i+k})
        # We want to minimize max(O_k)
        # Using soft max for differentiability
        # O_k = 50 - A_k
        # A_k = sum h_i h_{i+k}
        
        # Vectorized computation of A_k for k=1..K
        # h is length N
        # We can use np.roll
        A = np.zeros(K)
        for k in range(1, K + 1):
            h_shifted = np.roll(h, -k)
            A[k-1] = np.dot(h, h_shifted)
        
        O = 50.0 - A
        
        # Soft max of O
        # max(O) approx log(sum(exp(beta * O))) / beta
        beta = 50.0
        # Clip O to avoid overflow
        O_clipped = np.clip(O, -100, 100)
        max_val = np.max(O_clipped)
        exp_vals = np.exp(beta * (O_clipped - max_val))
        soft_max = max_val + np.log(np.sum(exp_vals)) / beta
        
        return soft_max

    def grad(h):
        # Gradient of soft max
        # d/dh (max O_k) approx sum_k p_k grad O_k
        # where p_k = softmax(beta * O_k)
        
        A = np.zeros(K)
        for k in range(1, K + 1):
            h_shifted = np.roll(h, -k)
            A[k-1] = np.dot(h, h_shifted)
        
        O = 50.0 - A
        
        beta = 50.0
        O_clipped = np.clip(O, -100, 100)
        max_val = np.max(O_clipped)
        exp_vals = np.exp(beta * (O_clipped - max_val))
        probs = exp_vals / np.sum(exp_vals)
        
        g = np.zeros(N)
        for k in range(1, K + 1):
            # O_k = 50 - sum h_i h_{i+k}
            # d O_k / d h_i = - (h_{i+k} + h_{i-k})
            # Note: sum h_i h_{i+k} = sum h_{i-k} h_i
            # So gradient is - (h_{i+k} + h_{i-k})
            h_plus = np.roll(h, -k)
            h_minus = np.roll(h, k)
            g -= probs[k-1] * (h_plus + h_minus)
            
        return g

    # Constraints
    # Sum h = 50
    # 0 <= h <= 1
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 50.0, 'jac': lambda h: np.ones(N)}
    ]
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Optimization
    start_time = time.time()
    deadline = 120.0
    margin = 5.0
    
    result = minimize(
        objective, 
        h0, 
        jac=grad, 
        method='L-BFGS-B', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    h_opt = result.x
    
    # Ensure constraints are met strictly
    # Project to sum=50 and [0,1]
    # Simple projection: shift and clip, then adjust
    for _ in range(10):
        h_opt = np.clip(h_opt, 0.0, 1.0)
        diff = 50.0 - np.sum(h_opt)
        if abs(diff) < 1e-9:
            break
        # Adjust by adding diff/N, then clip again
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0.0, 1.0)
        
    # Final check
    if np.sum(h_opt) != 50.0:
        # Fallback to constant if projection fails
        h_opt = np.full(N, 0.5)
        
    return h_opt
