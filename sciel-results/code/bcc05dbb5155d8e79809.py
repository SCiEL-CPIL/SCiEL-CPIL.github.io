import numpy as np
from scipy.optimize import differential_evolution
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
        diff = x_nodes[:, None] - roots[None, :]
        prod_diff = np.prod(diff**4, axis=1)
        integral_val = np.sum(w_nodes * prod_diff)
        return integral_val

    # Bounds: roots must be positive. 
    # Laguerre roots for m=12 are roughly between 1 and 40.
    # Let's use a wide range to allow the optimizer to find the global minimum.
    bounds = [(1e-6, 100) for _ in range(12)]
    
    # Use global optimization (Differential Evolution) to avoid local minima,
    # followed by a local polish step.
    result = differential_evolution(cost_func, bounds, maxiter=500, tol=1e-8, seed=42, polish=True)
    
    return list(result.x)
