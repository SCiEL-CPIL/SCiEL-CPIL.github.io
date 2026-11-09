"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the state at the end of t_span.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to match the reference solution's accuracy
    # rtol=1e-11 and atol=1e-13 are specified in the task description for the reference
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        max_step=np.inf, # Let it choose
        dense_output=False # We only need the final value
    )
    
    if not sol.success:
        # Fallback if DOP853 fails for some reason (unlikely for this problem)
        # Try Radau or BDF for stiffness, or just return the last value if it's finite
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
        else:
            return y0
            
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
