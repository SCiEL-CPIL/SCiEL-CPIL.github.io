import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal

def construct():
    """
    Optimizes the off-diagonal entries of a 12x12 tridiagonal matrix
    to minimize the variance of its eigenvalues (roots), which serves as
    a proxy for minimizing the uncertainty bound C4.
    Uses multiple restarts to avoid local minima.
    """
    N = 12
    # Fixed diagonal entries based on Laguerre recurrence (2i+1 for i=0..11)
    d = np.array([2*i + 1 for i in range(N)], dtype=float)
    
    def cost(params):
        o = params
        # Ensure off-diagonals are positive
        if np.any(o <= 0):
            return 1e10
        try:
            w, _ = eigh_tridiagonal(d, o, select='all')
        except Exception:
            return 1e10
            
        # Check positivity of roots
        if np.any(w <= 0):
            # Penalty for negative roots
            return 1e10 + np.sum(w[w <= 0])**2
            
        # Objective: Minimize variance (equivalent to minimizing sum of squared deviations from mean)
        mean_w = np.mean(w)
        var = np.sum((w - mean_w)**2)
        return var

    # Multiple restarts to find global minimum
    best_var = float('inf')
    best_o = None
    
    # Initial guesses
    guesses = [
        np.array([i for i in range(1, N)], dtype=float), # Standard Laguerre
        np.ones(N-1) * 5.0, # Uniform moderate
        np.ones(N-1) * 1.0, # Uniform small
        np.ones(N-1) * 10.0, # Uniform large
        np.random.uniform(0.5, 15.0, N-1) # Random
    ]
    
    for x0 in guesses:
        # Ensure x0 is within bounds
        x0 = np.clip(x0, 0.1, 100.0)
        
        bounds = [(0.1, 100.0) for _ in range(N-1)]
        
        try:
            res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-14})
            if res.success and res.fun < best_var:
                best_var = res.fun
                best_o = res.x
        except Exception:
            pass
        
    if best_o is not None:
        w, _ = eigh_tridiagonal(d, best_o, select='all')
        if np.all(w > 0):
            return [float(r) for r in w]
        
    # Fallback to standard Laguerre roots if optimization fails
    from scipy.special import roots_laguerre
    roots, _ = roots_laguerre(12)
    return [float(r) for r in roots]
