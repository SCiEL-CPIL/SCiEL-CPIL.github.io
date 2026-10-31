import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    alpha = 50.0
    
    def calc_c(h):
        """Calculate C_k for k=0..N-1"""
        cs = np.zeros(N)
        for k in range(N):
            if k == 0:
                r = np.sum(h * h)
            else:
                r = np.sum(h[k:] * h[:-k])
            cs[k] = (2.0 / N) * (N / 2.0 - r)
        return cs

    def objective(h):
        cs = calc_c(h)
        # Soft max
        m = np.max(cs)
        # Avoid overflow in exp
        # log_sum = log(sum(exp(alpha * (cs - m))))
        # exp_terms = exp(alpha * (cs - m))
        # Since cs - m <= 0, exp is <= 1.
        exp_terms = np.exp(alpha * (cs - m))
        log_sum = np.log(np.sum(exp_terms))
        soft_max = m + log_sum / alpha
        return soft_max

    def gradient(h):
        cs = calc_c(h)
        m = np.max(cs)
        exp_terms = np.exp(alpha * (cs - m))
        w = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros(N)
        # dC_k/dh_j = -2/N * (h_{j-k} + h_{j+k})
        # We sum over k: w_k * dC_k/dh_j
        
        # For each j, grad[j] = -2/N * sum_k w_k * (h_{j-k} + h_{j+k})
        # Note: h_i is 0 if i < 0 or i >= N
        
        # Efficiently compute this using convolution or loops
        # N is small (100), so loops are fine
        
        for k in range(N):
            if w[k] == 0:
                continue
            factor = -2.0 / N * w[k]
            for j in range(N):
                # Term 1: h_{j-k}
                idx1 = j - k
                if 0 <= idx1 < N:
                    grad[j] += factor * h[idx1]
                # Term 2: h_{j+k}
                idx2 = j + k
                if 0 <= idx2 < N:
                    grad[j] += factor * h[idx2]
                    
        return grad

    def constraint_sum(h):
        return np.sum(h) - N / 2.0

    constraints = [
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    bounds = [(0, 1)] * N
    
    # Run optimization
    # SLSQP with analytical gradient
    res = minimize(objective, h0, jac=gradient, method='SLSQP', bounds=bounds, 
                   constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
    
    if res.success:
        h_opt = res.x
    else:
        h_opt = h0
        
    # Ensure constraints are met strictly
    diff = N / 2.0 - np.sum(h_opt)
    if abs(diff) > 1e-9:
        mask = (h_opt > 0) & (h_opt < 1)
        if np.any(mask):
            h_opt[mask] += diff / np.sum(mask)
            h_opt = np.clip(h_opt, 0, 1)
            # Re-adjust if clipping changed sum
            diff2 = N / 2.0 - np.sum(h_opt)
            if abs(diff2) > 1e-9:
                h_opt = np.full(N, 0.5)
                
    return h_opt
