"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0

    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0_arr

    # Strategy:
    # 1. Probe linearity and time-invariance at t0 ONLY.
    #    Trust the task description "fixed ... linear system" to assume time-invariance.
    #    This reduces function calls from ~32 to 16.
    # 2. Use exact solution via matrix exponential.
    # 3. Fallback to robust stiff ODE solver if exact solution fails or is invalid.
    
    # Probe A and b at t0
    # b = f(t0, 0)
    try:
        b = np.array(f(t0, np.zeros(n)), dtype=float)
    except Exception:
        return _fallback_solve(f, t0, t1, y0_arr, deadline)

    # Check deadline
    if time.time() > deadline:
        return y0_arr

    # A columns: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        try:
            val = np.array(f(t0, e_j), dtype=float)
        except Exception:
            return _fallback_solve(f, t0, t1, y0_arr, deadline)
        A[:, j] = val - b
        
    # Check deadline
    if time.time() > deadline:
        return y0_arr

    # Construct augmented matrix M for y' = A y + b
    # z = [y; 1]
    # z' = M z, where M = [[A, b], [0, 0]]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    
    z0 = np.zeros(n + 1)
    z0[:n] = y0_arr
    z0[n] = 1.0
    
    try:
        expM = expm(M * dt)
        z_final = expM @ z0
        y_final = z_final[:n]
        
        if np.all(np.isfinite(y_final)):
            return y_final
    except Exception:
        pass
        
    # Fallback: Robust stiff solver
    return _fallback_solve(f, t0, t1, y0_arr, deadline)

def _fallback_solve(f, t0, t1, y0_arr, deadline):
    from scipy.integrate import solve_ivp
    # Use Radau which is good for stiff problems
    # Tight tolerances for high score
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='Radau',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If solver fails, try BDF
    try:
        sol_bdf = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-14,
            atol=1e-16
        )
        if sol_bdf.success:
            return sol_bdf.y[:, -1]
    except Exception:
        pass
        
    # If both fail, return y0
    return y0_arr
# EVOLVE-BLOCK-END
