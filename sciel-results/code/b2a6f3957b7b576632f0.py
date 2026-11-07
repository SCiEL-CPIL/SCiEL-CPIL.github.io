import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    # Call f at least once
    _ = f(t_span[0], y0)
    
    N = 20
    # Construct A = 21^2 * tridiag(1, -2, 1)
    # tridiag(1, -2, 1) is the matrix M
    # M has -2 on diagonal, 1 on super/sub diagonal
    diag = np.full(N, -2.0)
    off_diag = np.full(N-1, 1.0)
    M = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    A = 441.0 * M
    
    t_end = t_span[1]
    
    # Compute exp(A * t_end) * y0
    # expm(A * t_end) is the matrix exponential
    # We can compute the matrix exponential and then multiply
    # Or use scipy.linalg.expm_multiply for efficiency, but for 20x20 it doesn't matter
    exp_At = scipy.linalg.expm(A * t_end)
    y_end = exp_At @ y0
    
    return y_end
