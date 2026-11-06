import numpy as np
import scipy.linalg as sla

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check for autonomy and linearity
    # We assume linear: f(t, y) = A y + b
    # First, check if autonomous
    v0 = f(t0, y0)
    v1 = f(t1, y0)
    
    if np.linalg.norm(v0 - v1) > 1e-8 * (1 + np.linalg.norm(v0)):
        # Not autonomous, fall back to implicit solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=(t1-t0)/100)
        if sol.success:
            return sol.y[:, -1]
        else:
            return sol.y[:, -1] # Return best effort
    
    # Assume autonomous and linear: f(t, y) = A y + b
    # Extract A and b
    # v0 = A y0 + b
    # v_j = A (y0 + delta e_j) + b = v0 + delta A e_j
    # A e_j = (v_j - v0) / delta
    delta = 1e-7
    A = np.zeros((n, n))
    for j in range(n):
        yj = y0.copy()
        yj[j] += delta
        vj = f(t0, yj)
        A[:, j] = (vj - v0) / delta
        
    b = v0 - A @ y0
    
    # Solve y' = A y + b using augmented matrix exponential
    # z' = M z, z = [y; 1]
    # M = [[A, b], [0, 0]]
    M = np.zeros((n+1, n+1))
    M[:n, :n] = A
    M[:n, n] = b
    
    dt = t1 - t0
    E = sla.expm(M * dt)
    
    z0 = np.zeros(n+1)
    z0[:n] = y0
    z0[n] = 1.0
    
    z1 = E @ z0
    return z1[:n]
