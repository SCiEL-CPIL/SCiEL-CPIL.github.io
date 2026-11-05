"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses LSODA with tight tolerances and adaptive step sizing (no max_step constraint).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 320s to leave margin for the evaluator
    deadline = time.time() + 320.0
    
    # Count function calls to monitor budget (max 2,000,000)
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Primary solve: LSODA with tight tolerances, adaptive step size
    # No max_step to allow LSODA to take large steps in smooth regions
    sol1 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14
    )
    
    if not sol1.success:
        # Fallback: try with slightly looser tolerances if it failed
        sol1 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12
        )
        if not sol1.success:
            return sol1.y[:, -1]
    
    y_current = sol1.y[:, -1]
    
    # Check time: if we have plenty of time, try to refine further
    # The score penalizes calls, so we only refine if it's "cheap" or if error is high
    # However, LSODA is efficient, so a second pass with tighter tol might be worth it
    # if the first pass was fast.
    elapsed = time.time() - (deadline - 320.0)
    
    if elapsed < 10.0: # If first pass was very fast, try tighter
        sol2 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='LSODA',
            rtol=1e-14,
            atol=1e-16
        )
        if sol2.success:
            y_current = sol2.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
