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
    roots, _ = roots_laguerre(12)
    x0 = sorted(roots.tolist())
    
    # Pre-compute quadrature nodes and weights for the objective function
    # We integrate a polynomial of degree 48 (from (prod(x-xi)^2)^2) against e^{-x}
    # Gauss-Laguerre with N points is exact for degree 2N-1.
    # 2N - 1 >= 48 => N >= 24.5 => N = 25.
    # We use 50 for numerical stability and to handle potential slight deviations if the 
    # optimal solution isn't perfectly captured by the exact integration of the 
    # specific polynomial form (though it is a polynomial).
    quad_nodes, quad_weights = roots_laguerre(50)
    
    def objective(x):
        # x: array of 12 roots
        # Calculate P(x) = prod (x - xi)^2
        # diff shape: (50, 12)
        diff = quad_nodes[:, np.newaxis] - x[np.newaxis, :]
        # P_sq = prod (diff^2)
        P_sq = np.prod(diff**2, axis=1)
        
        # The integral is sum( weights * P_sq^2 )
        # Note: quad_weights from scipy already accounts for the e^{-x} weight in the 
        # definition of the quadrature rule for \int f(x)e^{-x}dx.
        val = np.sum(quad_weights * P_sq**2)
        
        # Penalty for non-positive or unordered roots to help optimization
        # We use bounds for positivity, but ordering is not a simple bound.
        # We add a small penalty if roots are not strictly increasing.
        penalty = 0.0
        for i in range(11):
            if x[i] >= x[i+1]:
                penalty += (x[i] - x[i+1])**2 * 1e5
                
        return val + penalty

    bounds = [(1e-8, None) for _ in range(12)]
    
    # Optimize
    # Use SLSQP which supports bounds
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-10})
    
    if result.success:
        optimal_roots = sorted(result.x)
    else:
        # If optimization fails, return the initial Laguerre roots
        optimal_roots = x0
        
    return optimal_roots.tolist()
