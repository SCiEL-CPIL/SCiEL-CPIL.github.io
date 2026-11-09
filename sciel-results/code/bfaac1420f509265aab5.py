import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) corresponding to the Cohn-Goncalves construction.
    These are the roots of the Laguerre polynomial L_12(x).
    """
    # m=12 is the degree of the Laguerre polynomial whose roots are the double roots
    # Use roots of L_12 as initial guess
    roots_init, _ = roots_laguerre(12)
    x0 = roots_init.tolist()
    
    # Setup Gauss-Laguerre quadrature nodes and weights for the integral
    # We need to integrate P(x)^2 e^{-x}. P is degree 24. P^2 is degree 48.
    # N=80 nodes should be sufficient for high accuracy.
    N_quad = 80
    x_quad, w_quad = roots_laguerre(N_quad)
    
    def objective(r):
        # r is a numpy array of 12 roots
        # Compute P(x) at quadrature points
        # P(x) = prod (x - r_i)^2
        # Using broadcasting
        diffs = x_quad[:, None] - r[None, :]
        # Square the differences and sum in log space to avoid overflow?
        # For now, direct multiplication. Roots are typically O(10).
        # (x-r)^2 can be large. 12 terms. 10^24 is fine in double.
        P_vals = np.prod(diffs**2, axis=1)
        
        # Integral approx: sum w_k * P(x_k)^2
        integral_val = np.sum(w_quad * P_vals)
        
        # P(0) = prod r_i^2
        P0 = np.prod(r**2)
        
        # Objective: sqrt(integral) / P0
        # We want to minimize this ratio.
        if P0 < 1e-10:
            return 1e10
        return np.sqrt(integral_val) / P0

    # Bounds for roots: positive
    # Heuristic bounds: [0.1, 50.0]
    bounds = [(0.1, 50.0)] * 12
    
    # Optimize
    # Using L-BFGS-B for efficiency with box constraints
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
    
    # Return the optimized roots
    opt_roots = res.x
    # Ensure they are positive (L-BFGS-B respects bounds, but just in case)
    opt_roots = np.clip(opt_roots, 1e-6, None)
    # Sort for consistency
    opt_roots = np.sort(opt_roots)
    
    return opt_roots.tolist()
