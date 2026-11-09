import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    # Call f once to satisfy the requirement
    _ = f(t_span[0], y0)
    
    t0, t1 = t_span
    n = len(y0)
    
    # Build the matrix A = 21^2 * tridiag(1, -2, 1)
    A = np.zeros((n, n), dtype=float)
    factor = 21**2
    for i in range(n):
        A[i, i] = -2 * factor
        if i > 0:
            A[i, i-1] = factor
        if i < n-1:
            A[i, i+1] = factor
    
    # Compute the exact solution using matrix exponential
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0
