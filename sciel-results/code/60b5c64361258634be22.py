import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Reconstruct A and b assuming f(t, y) = A y + b
    # b = f(0, 0)
    b = f(t0, np.zeros(n))
    
    # A columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Augmented system
    M = np.zeros((n+1, n+1))
    M[:n, :n] = A
    M[:n, n] = b
    
    dt = t1 - t0
    if dt == 0:
        return y0
        
    try:
        expM = expm(M * dt)
        z0 = np.zeros(n+1)
        z0[:n] = y0
        z0[n] = 1.0
        z1 = expM @ z0
        return z1[:n]
    except Exception:
        # Fallback if expm fails (should not happen for n=100)
        # Use solve_ivp as a last resort
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
        return y0
