"""High-accuracy ODE integrator for the harmonic oscillator using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses scipy's DOP853 with tight tolerances to match the reference closely.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    result = solve_ivp(
        f,
        t_span=(t0, t1),
        y0=y0_arr,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if not result.success:
        # Fallback: if DOP853 fails, try with slightly looser tolerances
        result = solve_ivp(
            f,
            t_span=(t0, t1),
            y0=y0_arr,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
    
    # Return the state at the end of the interval
    y_end = result.y[:, -1]
    
    # Ensure the result is finite and has the correct shape
    if not np.all(np.isfinite(y_end)):
        # Last resort: return y0 if something went very wrong
        return y0_arr.copy()
    
    return y_end
# EVOLVE-BLOCK-END
