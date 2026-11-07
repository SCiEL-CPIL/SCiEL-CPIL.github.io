import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _evaluate(h):
    """Compute the objective value for a given set of heights h."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)  # m = 2*N - 1 = 99
    y = np.zeros(m + 2)
    y[1:-1] = g  # y = [0, g[0], ..., g[m-1], 0]
    
    w = 1.0 / (m + 1)  # spacing on [-0.5, 0.5] with m+2 points
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _log_sum_exp(x, beta=1.0):
    """Differentiable approximation to max(x) using log-sum-exp."""
    # For large beta, this approaches max(x)
    max_x = np.max(x)
    if max_x < 0:
        # Handle negative values carefully
        scaled = beta * x
        m = np.max(scaled)
        if m > 700:
            return max_x
        return (1.0 / beta) * (m + np.log(np.sum(np.exp(scaled - m))))
    else:
        scaled = beta * x
        m = np.max(scaled)
        if m > 700:
            return max_x
        return (1.0 / beta) * (m + np.log(np.sum(np.exp(scaled - m))))

def _negative_value(h, beta=50.0):
    """Negative of the objective with differentiable max surrogate."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Use log-sum-exp surrogate for norm_inf
    abs_g = np.abs(g)
    norm_inf_approx = _log_sum_exp(abs_g, beta=beta)
    
    if norm_1 * norm_inf_approx < 1e-300:
        return 1e10
    
    return -l2 / (norm_1 * norm_inf_approx)

def _negative_value_grad(h, beta=50.0):
    """Negative of the objective with gradient (numerical)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10, np.zeros(N)
    
    val = _negative_value(h, beta=beta)
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        if h_minus[i] < 0:
            h_minus[i] = 0
        grad[i] = (_negative_value(h_plus, beta=beta) - _negative_value(h_minus, beta=beta)) / (h_plus[i] - h_minus[i])
    
    return val, grad

def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with uniform seed
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _evaluate(h0)
    
    # Try several beta values for the log-sum-exp surrogate
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - start_time > deadline - 30:
            break
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                fun=lambda x: _negative_value(x, beta=beta),
                x0=h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success and np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                val = _evaluate(result.x)
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Additional refinement with higher beta
    if time.time() - start_time < deadline - 20:
        try:
            result = minimize(
                fun=lambda x: _negative_value(x, beta=200.0),
                x0=best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success and np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                val = _evaluate(result.x)
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Ensure non-negativity and finiteness
    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
