import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use DOP853 with very tight tolerances to minimize relative error
    # This maximizes the -log10(rel) component of the score
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15
    )
    if not sol.success:
        # Fallback to RK45 if DOP853 fails (unlikely)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13
        )
    return sol.y[:, -1]
