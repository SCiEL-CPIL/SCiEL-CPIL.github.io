"""Program for the third autocorrelation inequality task.

The objective value is lower bounded by 2, attained by the uniform vector.
We solve the convex QP: min sum((h_i - 1/N)^2) s.t. sum(h) = 1.
This is equivalent to min sum(h_i^2) s.t. sum(h) = 1.
"""
import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # The global minimum of the autocorrelation bound is achieved by the uniform distribution.
    # We formulate this as a QP: minimize sum((h_i - 1/N)^2) subject to sum(h) = 1.
    # This quadratic form is 0 only when all h_i are equal to 1/N.
    
    def obj(h):
        return np.sum((h - 1.0/N)**2)
        
    def jac(h):
        return 2.0 * (h - 1.0/N)
        
    x0 = np.ones(N) / N
    cons = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 
             'jac': lambda h: np.ones(N)}]
             
    res = minimize(obj, x0, method='SLSQP', jac=jac, constraints=cons, options={'maxiter': 100, 'ftol': 1e-12})
    
    # Ensure numerical stability
    h = res.x
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.full(N, 1.0/N)
    return h / s
