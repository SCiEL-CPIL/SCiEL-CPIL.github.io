import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use Radau method for stiff problems, high accuracy
    # The reference uses Radau with rtol=1e-11, atol=1e-13
    # We use tighter tolerances to ensure rel error is minimized
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=0.1 # Optional: limit step size if needed, but Radau adapts well
    )
    
    if not sol.success:
        # Fallback to DOP853 if Radau fails (unlikely)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    return sol.y[:, -1]
