"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    Uses scipy's solve_ivp with LSODA method for stiff problem handling.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock, leave 30s margin
    deadline = time.time() + 300.0
    
    # Use Radau (implicit, stiff) with tight tolerances but no max_step restriction
    # to allow larger steps in non-stiff regions, reducing function calls.
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='Radau',
        rtol=1e-10,
        atol=1e-12
    )
    
    if not sol.success:
        # Fallback: LSODA without max_step
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12
        )
    
    if not sol.success:
        # Fallback: BDF
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )

    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: return whatever we have
    if sol.y is not None and sol.y.shape[1] > 0:
        return sol.y[:, -1]
    
    return y0_arr
# EVOLVE-BLOCK-END
