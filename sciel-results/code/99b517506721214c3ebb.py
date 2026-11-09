"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 290.0  # Safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Probe for linear time-invariant structure: y' = A*y + b
    # b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)

    # Verify linearity with a single test point
    y_test = np.ones(n) * 2.0
    expected = A @ y_test + b
    actual = np.array(f(t0, y_test), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-12):
        # Fallback: use scipy's solve_ivp with BDF method for stiff problems
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Check time-invariance at a different time
    t_mid = (t0 + t1) / 2
    expected_mid = A @ y_test + b
    actual_mid = np.array(f(t_mid, y_test), dtype=float)
    if not np.allclose(expected_mid, actual_mid, rtol=1e-10, atol=1e-12):
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Linear time-invariant system: y' = A*y + b
    # Solution: y(t) = expm(A*dt)*y0 + integral_0^dt expm(A*(dt-s))*b ds
    # The integral = (expm(A*dt) - I) @ A^{-1} @ b if A is invertible
    # More robust: use block structure or solve linear system
    
    M = expm(A * dt)
    
    # Compute the particular solution term: integral_0^dt expm(A*s) ds @ b
    # This equals (expm(A*dt) - I) @ A^{-1} @ b when A is invertible
    # Use solve with fallback to lstsq for singular A
    try:
        # Solve A @ X = (M - I) for X, then particular = X @ b
        # But this is equivalent to solving A @ p = (M - I) @ b for p directly
        rhs = (M - np.eye(n)) @ b
        particular = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # A is singular, use lstsq
        particular = np.linalg.lstsq(A, rhs, rcond=None)[0]
    
    y_result = M @ y0 + particular
    
    return y_result
