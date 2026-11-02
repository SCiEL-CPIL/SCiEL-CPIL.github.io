"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog, minimize

N = 600

# EVOLVE-BLOCK-START
def _compute_autoconv_max(h):
    """Compute the maximum of the discrete autoconvolution of h."""
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    Uses scipy.optimize.minimize to directly minimize max_k (h*h)[k]
    subject to sum(h) = 1, h >= 0.
    """
    # Objective: minimize max_k (h*h)[k]
    # Constraints: sum(h) = 1, h >= 0
    
    def objective(h):
        c = np.correlate(h, h, mode='full')
        return np.max(c)
    
    # Gradient of the objective (for SLSQP)
    def gradient(h):
        c = np.correlate(h, h, mode='full')
        k_max = np.argmax(c)
        # c[k_max] = sum_i h[i]*h[i+k_max] (with appropriate indexing)
        # The gradient with respect to h[j] is sum_i (delta_{i,j}*h[i+k_max] + delta_{i+k_max,j}*h[i])
        # For full correlation, c[k] = sum_i h[i]*h[i+k] for k>=0, and c[k] = sum_i h[i]*h[i+k] for k<0
        # Let's compute it numerically for safety
        eps = 1e-7
        grad = np.zeros_like(h)
        for j in range(N):
            h[j] += eps
            c_new = np.correlate(h, h, mode='full')
            grad[j] = (np.max(c_new) - c[np.argmax(c)]) / eps
            h[j] -= eps
        return grad
    
    # Use a better approach: since the problem is convex (minimize max of convex functions),
    # we can use SLSQP with analytical gradient
    
    def objective_and_grad(h):
        c = np.correlate(h, h, mode='full')
        idx = np.argmax(c)
        # For full mode, c has length 2N-1
        # c[idx] corresponds to lag = idx - (N-1)
        lag = idx - (N - 1)
        
        # c[lag] = sum_{i: 0<=i<N, 0<=i+lag<N} h[i]*h[i+lag]
        # d(c[lag])/dh[j] = sum_{i: i=j} h[i+lag] + sum_{i: i+lag=j} h[i]
        #                  = (h[j+lag] if 0<=j+lag<N else 0) + (h[j-lag] if 0<=j-lag<N else 0)
        
        grad = np.zeros(N)
        for j in range(N):
            term1 = h[j + lag] if 0 <= j + lag < N else 0.0
            term2 = h[j - lag] if 0 <= j - lag < N else 0.0
            grad[j] = term1 + term2
        
        return c[idx], grad
    
    # Start with uniform distribution
    h0 = np.ones(N) / N
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0,
         'jac': lambda h: np.ones(N)}
    ]
    
    # Bounds: h[i] >= 0
    bounds = [(0, None)] * N
    
    # Use SLSQP
    result = minimize(
        lambda h: objective_and_grad(h)[0],
        h0,
        jac=lambda h: objective_and_grad(h)[1],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )
    
    if result.success:
        h_opt = result.x
        # Ensure non-negativity and normalization
        h_opt = np.maximum(h_opt, 0)
        if h_opt.sum() > 0:
            h_opt = h_opt / h_opt.sum()
        else:
            h_opt = h0
    else:
        # Fallback to a good heuristic
        h_opt = h0
    
    return list(h_opt)
# EVOLVE-BLOCK-END
