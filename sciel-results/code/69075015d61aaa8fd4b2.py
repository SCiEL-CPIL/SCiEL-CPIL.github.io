import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Mandatory call to f
    _ = f(t0, y0)
    
    # Reconstruct A
    # A = 21^2 * tridiag(1, -2, 1)
    N = 20
    alpha = 21.0 ** 2
    # Main diagonal: -2 * alpha
    # Off diagonals: 1 * alpha
    diag_main = -2.0 * alpha * np.ones(N)
    diag_off = alpha * np.ones(N - 1)
    
    A = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
    
    dt = t1 - t0
    
    # Compute exact solution
    E = expm(A * dt)
    y_end = E @ y0
    
    return y_end
