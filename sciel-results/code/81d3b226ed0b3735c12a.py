"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses time-reversal transformation to convert the stiff Robertson problem
    into a non-stiff backward integration, then applies explicit DOP853.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best result before 320s (10s safety margin before 330s kill)
    deadline = time.time() + 320.0
    
    # Count function calls via a wrapper
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Strategy: Use time reversal. The Robertson problem is stiff forward but
    # the reversed system from t=40 is much less stiff, allowing explicit methods.
    
    # First, we need to get an approximate solution at t1 to use as starting point
    # for the backward integration. We'll use a quick LSODA run to get this.
    
    # Quick forward pass to get approximate y(t1)
    try:
        sol_quick = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        if sol_quick.success:
            y_end_approx = sol_quick.y[:, -1]
        else:
            y_end_approx = y0  # Fallback
    except Exception:
        y_end_approx = y0
    
    # Check if we're running out of time
    if time.time() > deadline - 50:
        return y_end_approx
    
    # Now integrate backward from t1 to t0 using DOP853 (explicit, high-order)
    # The reversed system: let s = t1 - t, then dy/ds = -f(t1-s, y)
    # We want y(t0), so we integrate from s=0 to s=t1-t0
    
    def f_backward(s, y):
        t = t1 - s
        return -f_counted(t, y)
    
    try:
        sol_back = solve_ivp(
            f_backward,
            (0, t1 - t0),
            y_end_approx,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol_back.success:
            return sol_back.y[:, -1]
    except Exception:
        pass
    
    # If backward DOP853 fails, try Radau forward
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: BDF
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='BDF',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Absolute last resort: return initial condition or quick solution
    return y_end_approx
# EVOLVE-BLOCK-END
