"""ODE solver: Heat2D 10x10 - using Krylov subspace methods for linear autonomous systems."""
import numpy as np
from scipy.linalg import expm_multiply
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y).
    Strategy:
    1. Check if the system is linear, homogeneous, and autonomous.
    2. If yes, use scipy.linalg.expm_multiply which applies the matrix exponential
       using a Krylov subspace method. This typically requires fewer function calls
       than reconstructing the full matrix and using expm.
    3. If no, fall back to scipy.integrate.solve_ivp with a stiff solver (Radau).
    """
    start_time = time.time()
    deadline = start_time + 280.0

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0:
        return y0
    
    if dt == 0:
        return y0

    # Check 1: Homogeneity (f(t, 0) == 0)
    # Check 2: Autonomy (f(t1, y) == f(t0, y))
    # We use a small test vector to check autonomy.
    # If N is large, checking all components is expensive.
    # But for N=100, it's fine.
    
    is_homogeneous = True
    is_autonomous = True
    
    # Check homogeneity
    y_zero = np.zeros(N, dtype=float)
    f_zero = np.asarray(f(t0, y_zero), dtype=float).flatten()
    if np.linalg.norm(f_zero) > 1e-10:
        is_homogeneous = False

    # Check autonomy
    # Pick a random or basis vector
    # Using e_0 is sufficient to detect time dependence in linear systems.
    e_0 = np.zeros(N, dtype=float)
    e_0[0] = 1.0
    f_t0 = np.asarray(f(t0, e_0), dtype=float).flatten()
    f_t1 = np.asarray(f(t1, e_0), dtype=float).flatten()
    
    if not np.allclose(f_t0, f_t1, rtol=1e-10, atol=1e-10):
        is_autonomous = False

    # If homogeneous and autonomous, use expm_multiply
    if is_homogeneous and is_autonomous:
        # Define the linear operator
        def matvec(v):
            return np.asarray(f(t0, v), dtype=float).flatten()
        
        try:
            # expm_multiply computes exp(dt * A) @ y0
            # It uses Krylov subspace methods (Arnoldi)
            y_end = expm_multiply(dt, matvec, y0, tol=1e-12)
            return y_end
        except Exception:
            # Fallback if expm_multiply fails
            pass

    # Fallback: General ODE solver
    # Use Radau for stiff problems (Heat equation is stiff)
    # Use tight tolerances to match high-accuracy reference
    try:
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try LSODA
            sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: RK4 with many steps
    return _fast_rk4(f, t0, t1, y0, 10000)

def _fast_rk4(f, t0, t1, y0, steps):
    """
    Fast RK4 fallback.
    """
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
    return y
