import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use DOP853, a high-order explicit Runge-Kutta method.
    # For the Van der Pol oscillator with mu=5, the problem is not highly stiff,
    # so a high-order explicit method can achieve high accuracy with significantly
    # fewer function evaluations than implicit methods like Radau.
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0_arr, 
            method='DOP853',
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau if DOP853 fails or is unstable
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0_arr, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return np.array([np.nan, np.nan])
