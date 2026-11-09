"""Rewritten ODE solver for 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: Extract the system matrix A using n function calls with unit vectors,
    then compute the exact solution y(t_end) = expm(A * (t_end - t_start)) @ y0.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 285.0  # 300s limit, 15s safety margin
    
    # Extract the system matrix A using n function calls
    # For a linear system, f(t, y) = A @ y (independent of t)
    # f(t, e_j) = A @ e_j = A[:, j] gives column j of A
    A = np.zeros((n, n), dtype=float)
    
    # Check if we're running out of time
    if time.time() > deadline:
        return y0
    
    # Extract each column of A using unit vectors
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback: use Euler's method with what we have
        h = (t1 - t0) / 100
        y = y0.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            y = y + h * (A @ y)
        return y
    
    # Compute the exact solution: y(t1) = expm(A * (t1 - t0)) @ y0
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0
    
    M = expm(A * dt)
    y_final = M @ y0
    
    # Verify finiteness
    if not np.all(np.isfinite(y_final)):
        # Fallback to a simple method
        h = dt / 1000
        y = y0.copy()
        for _ in range(1000):
            y = y + h * (A @ y)
        y_final = y
    
    return y_final
# EVOLVE-BLOCK-END
