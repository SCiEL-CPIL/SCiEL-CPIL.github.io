"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses adaptive implicit solvers (Radau) with tight tolerances for the Robertson problem.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Track elapsed time for deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for 330s limit
    
    def time_left():
        return deadline - time.time()
    
    # Strategy: Try Radau with progressively tighter tolerances within time budget
    # Radau is excellent for stiff problems like Robertson
    
    # First, try Radau with very tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.05
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau with tight tolerances failed or took too long, try BDF
    if time_left() > 10:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='BDF',
                rtol=1e-13,
                atol=1e-15,
                max_step=0.1
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Try LSODA as another stiff solver option
    if time_left() > 10:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='LSODA',
                rtol=1e-13,
                atol=1e-15,
                max_step=0.1
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback to Radau with slightly looser tolerances
    if time_left() > 5:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=0.1
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: simple RK4 with many steps, but check time
    h = (t1 - t0) / 50000
    t, y = t0, y0_arr.copy()
    n_steps = 50000
    for i in range(n_steps):
        if time_left() < 5:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
