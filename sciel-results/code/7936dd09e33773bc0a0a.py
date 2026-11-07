import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use DOP853 which is the reference method
    # Match tolerances to ensure high accuracy
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback to a robust method if DOP853 fails, though unlikely for this problem
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )
    return sol.y[:, -1]
