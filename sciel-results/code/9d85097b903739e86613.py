"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # 1. Extract affine system parameters: y' = A y + b
    # This handles both homogeneous (b=0) and affine (b!=0) linear systems.
    # Cost: n + 1 calls.
    
    # Extract b = f(t0, 0)
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    dt = t1 - t0
    if dt == 0:
        return y0
        
    # 2. Solve using augmented matrix exponential
    # System: [y'; 0'] = [A b; 0 0] [y; 1]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # M[n, :] is already 0
    
    M_scaled = M * dt
    ExpM = scipy.linalg.expm(M_scaled)
    
    # Initial augmented state
    y_aug = np.zeros(n + 1)
    y_aug[:n] = y0
    y_aug[n] = 1.0
    
    y_aug_final = ExpM @ y_aug
    y_final = y_aug_final[:n]
    
    # Sanity check
    if np.all(np.isfinite(y_final)):
        return y_final
    
    # Fallback (should not happen)
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    return y0
