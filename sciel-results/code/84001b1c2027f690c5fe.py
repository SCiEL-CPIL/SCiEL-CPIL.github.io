import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract matrix A assuming f(t, y) = A y
    # This requires n calls to f
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    # Compute matrix exponential
    # This is the most accurate method for linear systems
    M = scipy.linalg.expm(A * dt)
    y_end = M @ y0
    
    return y_end
