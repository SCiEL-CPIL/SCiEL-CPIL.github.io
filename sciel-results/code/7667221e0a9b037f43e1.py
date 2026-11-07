"""Rewritten program using adaptive high-order Runge-Kutta (DOP853) via SciPy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the ODE system using DOP853 with tight tolerances matching the reference.
    """
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # In case of failure, raise an exception as per task rules (invalid result)
        raise RuntimeError(f"Integration failed: {sol.message}")
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
