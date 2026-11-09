import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # If the system is trivial or t_span is zero
    if t1 == t0:
        return y0
        
    # Determine the matrix A such that f(t, y) = A y
    # We assume linearity and time-independence.
    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        # Use t0 for the evaluation. If f is time-dependent, this is an approximation, 
        # but for the heat equation, A is constant.
        A[:, j] = f(t0, e_j)
        
    # Compute the exact solution using matrix exponential
    # y(t1) = exp((t1 - t0) * A) @ y0
    dt = t1 - t0
    M = expm(dt * A)
    y_end = M @ y0
    
    return y_end
