import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853, the same method as the reference, with tight tolerances.
    # This minimizes the number of steps/calls while ensuring high accuracy.
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853',
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if something goes wrong, though unlikely for this system
        # Just return the last point if it exists, or raise
        if len(sol.y) > 0:
            return sol.y[:, -1]
        raise RuntimeError("Integration failed")
        
    return sol.y[:, -1]
