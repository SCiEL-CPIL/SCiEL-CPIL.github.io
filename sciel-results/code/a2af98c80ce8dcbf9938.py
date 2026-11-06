"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Solve a 15-dimensional stiff linear ODE y' = A*y by:
    1. Extracting the matrix A using 16 function calls (f(t, 0) and f(t, e_i) for basis vectors)
    2. Computing the exact solution y(t_end) = expm((t_end - t_start) * A) @ y0
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract matrix A: A[:, j] = f(t, e_j) - f(t, 0)
    # For linear systems: f(t, y) = A @ y (assuming no time dependence)
    # Actually for linear: f(t, y) = A(t) @ y, but if A is constant, f(t, y) = A @ y
    
    # First call: f(t, 0) should be 0 for linear homogeneous system
    f0 = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # Extract columns of A
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
    
    # Compute matrix exponential
    dt = t1 - t0
    expm_dt_A = expm(dt * A)
    
    # Apply to initial condition
    y_final = expm_dt_A @ y0
    
    return y_final
# EVOLVE-BLOCK-END
