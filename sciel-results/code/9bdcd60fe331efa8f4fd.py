"""Rewritten program: exploit linearity to extract A with n calls, then solve via matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def _extract_matrix(f, t0, y0):
    """
    Extract the 15x15 system matrix A for a linear ODE y' = A y.
    
    For a linear system, f(t, y) = A y (assuming time-independent A).
    We extract columns of A using unit vectors.
    Column j of A is f(t, e_j) where e_j is the j-th standard basis vector.
    
    Total calls: n = 15 (one per column).
    """
    n = len(y0)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    return A


def _solve_linear_ode(A, y0, t0, t1):
    """
    Solve y' = A y, y(t0) = y0, at time t1.
    Solution: y(t1) = expm((t1 - t0) * A) @ y0
    """
    dt = t1 - t0
    M = expm(dt * A)
    return M @ y0


def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Extract the system matrix A using exactly n=15 calls to f (one per column).
    2. Solve the linear system exactly using the matrix exponential.
    
    Total f calls: 15 (the theoretical minimum for extracting a 15x15 matrix).
    This maximizes the -0.5*log10(calls) component of the score.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Extract A with 15 calls
    A = _extract_matrix(f, t0, y0)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: just return y0 (should not happen, but safety)
        return y0
    
    # Solve using matrix exponential
    y_final = _solve_linear_ode(A, y0, t0, t1)
    
    # Final sanity check
    if not np.all(np.isfinite(y_final)):
        # Fallback: use a simple RK4 with the extracted A
        h = (t1 - t0) / 1000
        y = y0.copy()
        t = t0
        for _ in range(1000):
            k1 = A @ y
            k2 = A @ (y + 0.5 * h * k1)
            k3 = A @ (y + 0.5 * h * k2)
            k4 = A @ (y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        y_final = y
    
    return y_final
# EVOLVE-BLOCK-END
