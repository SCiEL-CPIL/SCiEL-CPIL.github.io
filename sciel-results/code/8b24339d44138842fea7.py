import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # LSODA is often more efficient for stiff problems like Van der Pol
    # It automatically switches between Adams and BDF methods.
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='LSODA', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback to Radau if LSODA fails
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
    return sol.y[:, -1]
