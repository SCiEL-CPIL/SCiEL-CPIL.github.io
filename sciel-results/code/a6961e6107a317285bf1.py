import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances to maximize accuracy
    # while keeping the call count low due to high order.
    # The reference uses rtol=1e-11, atol=1e-13.
    # Using tighter tolerances (1e-15) should drive rel error below 1e-14.
    # DOP853 is 11th order, so it requires few calls.
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
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
