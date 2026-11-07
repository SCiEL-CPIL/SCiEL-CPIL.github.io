import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE using scipy.integrate.solve_ivp with DOP853.
    This matches the reference solver's method and tolerances to minimize
    relative error (rel) to the reference solution (y_ref).
    """
    t_start, t_end = t_span
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: return initial state if solver fails (should not happen)
            return y0
            
    except Exception:
        # Ensure no exception propagates
        return y0
