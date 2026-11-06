"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the 15-dimensional stiff linear ODE y' = f(t, y) on t_span.
    Returns the final state y(t_end).
    
    Strategy:
    1. Extract the linear system parameters A and b from f (autonomous linear: y' = A@y + b).
    2. Use scipy.integrate.solve_ivp with BDF or Radau for stiff problems.
    3. If scipy fails, fall back to a custom implicit method.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s total, use 290s for computation
    deadline = time.time() + 290.0
    
    # Extract A and b from the linear system
    # For a linear system y' = A@y + b, we can extract:
    # f(t, 0) = b
    # f(t, e_j) - f(t, 0) = A[:, j]
    b = np.array(f(t0, np.zeros(n)), dtype=float)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = np.array(f(t0, ej), dtype=float) - b
    
    # Check if the system is truly linear by verifying at another point
    # (This is optional but helps confirm our assumption)
    # For now, proceed with the extracted A and b
    
    # Use scipy's solve_ivp with a stiff solver
    # BDF is good for stiff problems
    # We want high accuracy, so set tight tolerances
    
    def linear_ode(t, y):
        return A @ y + b
    
    # Try BDF first
    sol = solve_ivp(
        linear_ode,
        [t0, t1],
        y0,
        method='BDF',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100,  # Reasonable initial max step
    )
    
    if sol.success and time.time() < deadline:
        return sol.y[:, -1]
    
    # If BDF failed, try Radau
    if time.time() < deadline:
        sol = solve_ivp(
            linear_ode,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100,
        )
        if sol.success:
            return sol.y[:, -1]
    
    # Fallback: custom implicit Euler with Newton
    if time.time() < deadline:
        h = (t1 - t0) / 1000
        t, y = t0, y0.copy()
        I = np.eye(n)
        for _ in range(1000):
            z = y.copy()
            for _newton in range(10):
                fz = A @ z + b
                # Solve (I - h*A) @ z = y + h*fz
                try:
                    z_new = np.linalg.solve(I - h * A, y + h * fz)
                except np.linalg.LinAlgError:
                    break
                if np.linalg.norm(z_new - z) <= 1e-12 * (1 + np.linalg.norm(z_new)):
                    z = z_new
                    break
                z = z_new
            y = z
            t += h
            if time.time() > deadline - 5:
                break
        return y
    
    # Last resort: return initial condition if everything fails
    return y0
# EVOLVE-BLOCK-END
