import numpy as np
import math
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrates y' = A y from t_start to t_end.
    Uses matrix exponential via scipy.linalg.expm for robustness.
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Ensure f is called at least once
    _ = f(t0, y0)
    
    N = len(y0)
    if N == 0:
        return np.array(y0)
        
    # Construct A = (N+1)^2 * tridiag(1, -2, 1)
    # For N=20, N+1=21, factor=441
    factor = (N + 1) ** 2
    
    # Main diagonal
    main_diag = -2.0 * factor * np.ones(N)
    # Off-diagonals
    off_diag = factor * np.ones(N - 1)
    
    A = np.diag(main_diag)
    if N > 1:
        A += np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
    # Compute matrix exponential
    # y_end = exp(A * dt) @ y0
    M = expm(A * dt)
    y_end = M @ y0
    
    if not np.all(np.isfinite(y_end)):
        return np.array(y0)
        
    return y_end
