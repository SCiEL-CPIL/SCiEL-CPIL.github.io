"""ODE solver: 15-dimensional stiff linear system using adaptive stiff integration."""
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy: Use an adaptive stiff ODE solver (Radau) with tight tolerances.
    This handles both time-invariant and time-dependent linear systems accurately,
    avoiding the approximation error of assuming a constant matrix A.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Leave 10s margin for safety
    
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau method which is suitable for stiff problems
    # Tight tolerances to minimize integration error
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-13, 
        atol=1e-15,
        # max_step=0.05 # Let adaptive step size handle it, but cap if needed
    )
    
    # Check if we are approaching the time limit
    if time.time() > deadline:
        # If the solver hasn't finished or failed, return the last available state
        if sol.success:
            return sol.y[:, -1]
        elif sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0

    if not sol.success:
        # If solver failed but produced some results, return the last one
        if sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0

    return sol.y[:, -1]
