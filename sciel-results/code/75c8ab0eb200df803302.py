import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use DOP853 which is a high-order (8th order) Runge-Kutta method
    # with adaptive step size.
    # Tight tolerances to minimize error relative to the reference.
    # The reference uses rtol=1e-11, atol=1e-13.
    # We use slightly tighter tolerances to ensure we are as close as possible,
    # but DOP853 is efficient so call count should remain low.
    
    # Add a safety deadline check? 
    # solve_ivp is usually very fast for this small system.
    # We can just run it.
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=(t1-t0)/10 # Prevent huge steps, though adaptive will handle it
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback if it fails, though unlikely
        # Use a simpler method with more steps
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )
        return sol.y[:, -1]
