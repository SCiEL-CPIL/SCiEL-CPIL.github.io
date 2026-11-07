<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_optimal_depth(C, alpha, k):
    """
    Finds optimal depth x >= 0 maximizing (1 - exp(-alpha * exp(-k*x))) * (x + C)
    C is the value difference V(q+1) - V(q) for bid, or V(q-1) - V(q) for ask.
    """
    if alpha <= 0:
        return 0.0
    
    # If C is very large positive, we want to maximize fill probability to capture C?
    # Actually, if C > 0, (x+C) is large.
    # If C < 0, we might want to avoid fills?
    
    def neg_obj(x):
        if x < 0: return -1e10
        p = 1.0 - np.exp(-alpha * np.exp(-k * x))
        return -p * (x + C)
        
    # Initial guess
    # If C is large, x should be small to get high p? Or large to get high x?
    # If C is negative, we want p to be small (x large) or x small?
    # Let's use a bounded scalar minimization.
    # Upper bound: if x is too large, p->0, obj->0.
    # If x is 0, obj = (1-exp(-alpha)) * C.
    # Max x: 10/k is usually enough for p to be negligible.
    x_max = 10.0 / k
    
    res = minimize_scalar(neg_obj, bounds=(0, x_max), method='bounded')
    return res.x
=======
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0
_POLICY_CACHE = {}

def _solve_optimal_depth_vec(C_arr, alpha, k):
    """
    Vectorized solution for optimal depth x >= 0 maximizing
    f(x) = (1 - exp(-alpha * exp(-k*x))) * (x + C)
    
    C_arr: array of shape (N,) with C = V_next[q±1] - V_next[q]
    Returns array of optimal depths.
    """
    N = len(C_arr)
    result = np.zeros(N, dtype=np.float64)
    
    # For C >= 0: f(0) = (1-exp(-alpha))*C >= 0, and f(x) -> 0 as x->inf
    # So optimal is at x=0 (or a local max). Let's check if x=0 is optimal.
    # f'(x) at x=0: derivative of p(x)*(x+C) at x=0
    # p(x) = 1 - exp(-alpha*exp(-kx))
    # p'(x) = alpha*exp(-kx)*(-k)*(-exp(-kx))... wait
    # d/dx exp(-kx) = -k*exp(-kx)
    # d/dx [1 - exp(-alpha*exp(-kx))] = -exp(-alpha*exp(-kx)) * (-alpha) * (-k) * exp(-kx)
    # = -alpha * k * exp(-kx) * exp(-alpha*exp(-kx))
    # At x=0: p'(0) = -alpha*k*exp(-alpha)
    # f'(x) = p'(x)*(x+C) + p(x)*1
    # f'(0) = -alpha*k*exp(-alpha)*C + (1-exp(-alpha))
    # If f'(0) <= 0, then x=0 is optimal (f is decreasing from 0).
    # If f'(0) > 0, there might be an interior max.
    
    # For C < 0: f(0) = (1-exp(-alpha))*C < 0
    # f(x) -> 0 as x->inf. So we want to maximize (make less negative or positive).
    # The optimal might be at x=0 or at some interior point where f crosses 0 and peaks.
    
    # Let's handle all cases numerically but efficiently.
    # Use a grid search approach with refinement, or solve the ODE-like condition.
    
    # Actually, let's just use a vectorized bisection/Newton on the first-order condition.
    # f'(x) = p'(x)*(x+C) + p(x) = 0
    # where p(x) = 1 - exp(-alpha*exp(-kx))
    # p'(x) = -alpha*k*exp(-kx)*exp(-alpha*exp(-kx))
    
    # Let u = alpha*exp(-kx), so x = -ln(u/alpha)/k
    # p = 1 - exp(-u)
    # p' = -u' * exp(-u) = -(du/dx)*exp(-u) = -(-k*u)*exp(-u) = k*u*exp(-u)
    # Wait: du/dx = -k*u, so p' = -exp(-u)*(-k*u) = k*u*exp(-u)
    
    # f'(x) = k*u*exp(-u)*(x+C) + (1-exp(-u)) = 0
    # x = -ln(u/alpha)/k
    # k*u*exp(-u)*(-ln(u/alpha)/k + C) + 1 - exp(-u) = 0
    # u*exp(-u)*(-ln(u/alpha) + k*C) + k*(1-exp(-u)) = 0
    
    # This is complex. Let's just do a simple vectorized grid search.
    # x in [0, x_max] where x_max = 15/k (p is negligible beyond this)
    x_max = 15.0 / k
    # Use 200 grid points
    x_grid = np.linspace(0, x_max, 200)
    # Compute f(x) for all C values at all grid points
    # f(x, C) = (1 - exp(-alpha*exp(-k*x))) * (x + C)
    # Shape: (200, N)
    exponents = alpha * np.exp(-k * x_grid)  # shape (200,)
    probs = 1.0 - np.exp(-exponents)  # shape (200,)
    # f[i, j] = probs[i] * (x_grid[i] + C_arr[j])
    f_vals = probs[:, np.newaxis] * (x_grid[:, np.newaxis] + C_arr[np.newaxis, :])  # (200, N)
    
    # Find argmax along axis 0
    best_idx = np.argmax(f_vals, axis=0)
    result = x_grid[best_idx]
    
    # Refine: for each point, do a local search around the best grid point
    # This is still O(N * local_points) which is fine for N=61
    for i in range(N):
        # Local refinement around best_idx[i]
        lo = max(0, best_idx[i] - 2)
        hi = min(199, best_idx[i] + 2)
        if hi - lo < 3:
            continue
        local_x = np.linspace(x_grid[lo], x_grid[hi], 20)
        local_f = (1.0 - np.exp(-alpha * np.exp(-k * local_x))) * (local_x + C_arr[i])
        local_best = local_x[np.argmax(local_f)]
        # Also check if x=0 is better (boundary)
        f0 = (1.0 - np.exp(-alpha)) * C_arr[i]
        if f0 > np.max(local_f):
            result[i] = 0.0
        else:
            result[i] = local_best
    
    return result
>>>>>>> REPLACE

