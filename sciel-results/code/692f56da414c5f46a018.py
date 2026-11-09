"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra ODE system from t_span[0] to t_span[1]
    starting at y0, and return the state at t_span[1].
    
    Uses scipy's DOP853 with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to match the reference
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        max_step=(t1 - t0) / 100  # reasonable max step
    )
    
    if not sol.success:
        # Fallback: try with even tighter settings or different method
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 200
        )
    
    if sol.success and len(sol.t) > 0:
        return sol.y[:, -1]
    
    # Last resort: return initial condition (shouldn't happen)
    return y0_arr
# EVOLVE-BLOCK-END
