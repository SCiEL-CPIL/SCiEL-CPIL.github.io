import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # We formulate the problem as maximizing ||h||_2^2 subject to:
    # 1. sum(h) = 1
    # 2. h >= 0
    # 3. max_k (h*h)_k <= 1
    #
    # This is a Quadratic Programming problem.
    # We use scipy.optimize.minimize with SLSQP.
    
    # Objective: Minimize -sum(h^2)
    def objective(h):
        return -np.sum(h**2)
    
    # Constraints
    # Equality: sum(h) - 1 = 0
    # Inequality: 1 - (h*h)_k >= 0 for all k
    
    cons = []
    
    # Equality constraint
    cons.append({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    
    # Inequality constraints for autoconvolution peaks
    # (h*h)_k = sum_i h_i h_{i+k}
    # We need to define the function for each k.
    # There are 2N - 1 lags.
    # To speed up, we can precompute the index pairs for each lag.
    # Or just use convolve inside the constraint? That might be slow.
    # Better to precompute masks or use matrix form.
    
    # Let's use a matrix A such that (A h)_k = (h*h)_k?
    # (h*h)_k = sum_i h_i h_{i+k}. This is quadratic.
    # We can't easily represent as linear constraints.
    # But minimize supports nonlinear constraints.
    
    # We will define the constraint function to compute the max violation.
    # Actually, we need individual constraints for SLSQP to work well.
    # Or we can use a single constraint: max_k ((h*h)_k - 1) <= 0.
    # SLSQP handles scalar constraints.
    
    def autoconv_max_constraint(h):
        h_conv = np.convolve(h, h, mode='full')
        return np.max(h_conv) - 1.0
    
    cons.append({'type': 'ineq', 'fun': lambda h: 1.0 - np.max(np.convolve(h, h, mode='full'))})
    
    # Bounds
    bounds = [(0, 1) for _ in range(N)]
    
    # Initial guess: Uniform
    h0 = np.ones(N) / N
    
    # Run optimization
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            h_opt = res.x
            # Ensure non-negativity and normalization
            h_opt = np.clip(h_opt, 0, None)
            h_opt = h_opt / np.sum(h_opt)
            return list(h_opt)
        else:
            # Fallback to uniform if optimization fails
            return list(h0)
    except Exception:
        return list(h0)

