import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.laguerre import laggauss

def construct():
    # We want to minimize the "upper bound on C4".
    # Hypothesis: The score is proportional to the L4 norm of the polynomial 
    # formed by the double roots, weighted by the Laguerre weight e^-x.
    # i.e., minimize I = int_0^inf e^-x * (prod (x-xi)^2)^2 dx = int e^-x * prod (x-xi)^4 dx
    
    # Use Gaussian Quadrature to compute the integral
    # Number of quadrature points should be large enough to resolve the polynomial
    # Degree of integrand is 4 * 12 = 48. 
    # Laguerre quadrature of order N is exact for polynomials up to degree 2N-1.
    # So we need 2N-1 >= 48 => N >= 24.5 => N=25 is minimum. 
    # Let's use N=100 for safety and accuracy.
    n_quad = 100
    x_nodes, w_nodes = laggauss(n_quad)
    
    def cost_func(roots):
        # roots is a 1D array of 12 values
        # Calculate prod (x - xi)^4
        # To avoid overflow/underflow, we can work with logs or just compute directly if x_nodes are reasonable.
        # x_nodes for N=100 range from ~0.05 to ~400.
        # The product might be large.
        # Let's compute the integral directly.
        # integrand = w_nodes * exp(-x_nodes) is already in w_nodes for laggauss?
        # No, laggauss returns nodes and weights for int_0^inf e^-x f(x) dx.
        # So integral = sum(w_nodes * f(x_nodes))
        
        # f(x) = prod (x - xi)^4
        # We can compute this as (prod (x - xi)^2)^2
        # Or just compute term by term.
        
        # Vectorized calculation:
        # roots shape (12,), x_nodes shape (100,)
        # diff = x_nodes[:, None] - roots[None, :]  # shape (100, 12)
        # prod_diff = np.prod(diff**4, axis=1)
        # integral = np.sum(w_nodes * prod_diff)
        
        # To prevent overflow, we might scale. But let's try direct first.
        diff = x_nodes[:, None] - roots[None, :]
        # If any root is very large, diff can be large.
        # (x-xi)^4 can be huge.
        # Let's use a log-sum-exp trick or just scale the roots?
        # Actually, for m=12, roots are around 10-20.
        # x_nodes go up to ~400.
        # (400-20)^4 is ~ 2.5e10.
        # Sum of 100 terms... ~ 1e12.
        # This fits in float64.
        
        prod_diff = np.prod(diff**4, axis=1)
        integral_val = np.sum(w_nodes * prod_diff)
        
        # The problem asks for the "upper bound on C4".
        # If the score is this integral, lower is better.
        # If the score is something else derived from this, minimizing this integral is a safe bet.
        
        return integral_val

    # Initial guess: Roots of L_12(x)
    x_init, _ = laggauss(12)
    
    # Bounds: roots must be positive
    bounds = [(1e-6, None) for _ in range(12)]
    
    # Optimize
    result = minimize(cost_func, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    
    if result.success:
        return list(result.x)
    else:
        # Fallback to initial guess if optimization fails
        return list(x_init)
