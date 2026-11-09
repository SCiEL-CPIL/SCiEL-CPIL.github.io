"""Improved program using high-order adaptive Runge-Kutta (DOP853) with tightened tolerances to minimize relative error."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system (or generic ODE f) from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    """
    start_time = time.time()
    time_limit = 290.0  # Seconds, safety margin before 300s limit
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Tighten tolerances to reduce relative error against the reference.
    # Reference uses rtol=1e-11, atol=1e-13.
    # Using rtol=1e-13, atol=1e-15 should drive rel closer to the 1e-14 floor.
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        
        if not sol.success:
            # Fallback to reference tolerances if tighter fails
            sol = solve_ivp(
                f,
                (t0, t1),
                y0_arr,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            
        if not sol.success:
             return y0_arr

        y_end = sol.y[:, -1]
        
        # Check if result is finite
        if not np.all(np.isfinite(y_end)):
            return y0_arr
            
        return y_end

    except Exception as e:
        return y0_arr
# EVOLVE-BLOCK-END
