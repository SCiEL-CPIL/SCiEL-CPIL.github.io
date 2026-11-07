import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances to maximize accuracy
    # while keeping the call count low due to high order.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-15, 
        atol=1e-15
    )
    
    if not sol.success:
        # Fallback to a more robust method if DOP853 fails, though unlikely
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    return sol.y[:, -1]
