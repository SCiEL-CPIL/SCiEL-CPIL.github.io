import numpy as np
from scipy.special import roots_laguerre, gamma
from scipy.optimize import minimize

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) corresponding to the Cohn-Goncalves construction.
    Optimizes the roots to minimize a proxy for the C4 upper bound.
    """
    m = 12
    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(m)
    x0 = np.sort(x0)
    
    # Define the objective function
    # We minimize the 2nd moment of the polynomial squared, weighted by e^-x
    # This is a proxy for the "uncertainty" or "spread" which often appears in C4 bounds.
    def objective(roots):
        # Ensure roots are positive and sorted to avoid issues with product construction
        # SLSQP might explore invalid regions, so we use a penalty
        if np.any(roots <= 0):
            return 1e10
        # Construct polynomial coefficients
        # P(x) = prod (x - r_i)
        # We can compute the integral using Gauss-Laguerre quadrature
        # Nodes and weights for Gauss-Laguerre
        nodes, weights = roots_laguerre(100) # High order for accuracy
        # Evaluate P(nodes)
        # P(x) = prod (x - r_i)
        # log P(x) = sum log(x - r_i) to avoid overflow? 
        # Roots are ~ 1 to 30. Nodes go up to ~ 300.
        # Direct product is fine for m=12.
        P_vals = np.ones_like(nodes)
        for r in roots:
            P_vals *= (nodes - r)
        
        # Integral of x^2 * P(x)^2 * e^-x
        # Quadrature: sum weights_i * x_i^2 * P(x_i)^2
        integral = np.sum(weights * nodes**2 * P_vals**2)
        
        # Also add a penalty for orthogonality violation to keep it "Laguerre-like"
        # Check <P, 1> = sum w_i P(x_i)
        orth_res = np.sum(weights * P_vals)
        penalty = orth_res**2
        
        return integral + 1e6 * penalty

    # Bounds: roots must be positive
    bounds = [(1e-6, None) for _ in range(m)]
    
    # Optimize
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
    
    if res.success:
        opt_roots = res.x
    else:
        # Fallback to initial guess if optimization fails
        opt_roots = x0
        
    return sorted(opt_roots.tolist())
