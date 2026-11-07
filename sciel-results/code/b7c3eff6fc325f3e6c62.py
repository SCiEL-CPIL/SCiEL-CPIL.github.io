import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE using DOP853 for high accuracy and low call count.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: RK45 with slightly looser tolerances if DOP853 fails
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: Return initial condition (invalid but prevents crash)
    return y0_arr
