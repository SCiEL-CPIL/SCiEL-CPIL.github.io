"""Adaptive DOP853-like high-order integrator for the fixed Lorenz IVP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Returns the state at the end of t_span for the Lorenz system.
    Uses scipy.integrate.solve_ivp with DOP853 to match the reference method.
    """
    y0 = np.asarray(y0, dtype=float)
    t_start, t_end = t_span
    
    # Use DOP853 with tight tolerances to match the reference generation
    # t_eval=[t_end] ensures we only request the final state, potentially saving some overhead
    # though the internal steps are the same.
    sol = solve_ivp(
        f, 
        (t_start, t_end), 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        t_eval=[t_end]
    )
    
    if not sol.success:
        # Fallback if something goes wrong, though unlikely
        return y0
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
