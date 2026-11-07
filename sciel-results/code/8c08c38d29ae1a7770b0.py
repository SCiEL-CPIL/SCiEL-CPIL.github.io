import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances
    # rtol=1e-13, atol=1e-15 should give high accuracy
    # max_step can be set to ensure resolution, but adaptive is usually fine
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=(t1 - t0) / 100.0  # Limit step size to prevent large errors in oscillatory regions
    )
    
    if not sol.success:
        # Fallback to a simpler method if DOP853 fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0
        )
        
    return sol.y[:, -1]
