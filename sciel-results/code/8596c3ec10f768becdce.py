"""High-accuracy ODE integrator for the Lorenz system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 330s kill
    
    # First attempt: use DOP853 with tight tolerances matching the reference
    # This should give very small relative error with moderate call count
    result = solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    elapsed = time.time() - start_time
    
    if result.y is not None and result.y.shape[1] > 0:
        y_end = result.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
    
    # Fallback: if something went wrong, try with slightly looser tolerances
    if time.time() < deadline:
        result2 = solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
        if result2.y is not None and result2.y.shape[1] > 0:
            y_end = result2.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    
    # Last resort: return y0 (shouldn't happen)
    return y0
# EVOLVE-BLOCK-END
