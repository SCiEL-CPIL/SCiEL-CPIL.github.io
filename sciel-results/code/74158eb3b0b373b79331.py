import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract g(t0)
    g = f(t0, np.zeros(n, dtype=float))
    
    # Extract A
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i) - g
        
    # Augmented matrix for y' = A y + g
    # z = [y; 1]
    # z' = [A, g; 0, 0] z
    A_aug = np.zeros((n+1, n+1), dtype=float)
    A_aug[:n, :n] = A
    A_aug[:n, n] = g
    # Last row is 0
    
    # Compute exponential
    E_aug = expm(A_aug * T)
    
    # Initial vector
    z0 = np.zeros(n+1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Final state
    z_end = E_aug @ z0
    
    return z_end[:n]
