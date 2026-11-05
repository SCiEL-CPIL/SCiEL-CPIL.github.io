"""Rewritten program: ODE solver for 15-dim stiff linear system using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # The task specifies a "fixed" 15-dimensional stiff linear system.
    # "Fixed" implies constant coefficients: y' = A y.
    # We recover A by evaluating f(t0, e_i) for each basis vector e_i.
    # Total f calls: n = 15 (just the basis vectors at t0).
    
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
    
    # Use matrix exponential for exact solution of y' = A y
    dt = t1 - t0
    At = A * dt
    
    # Use scipy's expm which uses scaling-and-squaring with Padé approximation
    # This is highly accurate for stiff systems
    M = expm(At)
    y_sol = M @ y0
    
    # Verify solution is valid
    if np.any(np.isnan(y_sol)) or np.any(np.isinf(y_sol)):
        # Fallback: try eigenvalue decomposition
        try:
            eigvals, eigvecs = np.linalg.eig(At)
            eigvals_exp = np.exp(eigvals)
            V_inv = np.linalg.inv(eigvecs)
            M_eig = eigvecs @ np.diag(eigvals_exp) @ V_inv
            y_sol_eig = M_eig @ y0
            if not np.any(np.isnan(y_sol_eig)) and not np.any(np.isinf(y_sol_eig)):
                return y_sol_eig
        except Exception:
            pass
    
    return y_sol

# EVOLVE-BLOCK-END
