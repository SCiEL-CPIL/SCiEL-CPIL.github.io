import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use DOPRI5 (Dormand-Prince 5th order) which is explicit and does not require
    # Jacobian evaluations. For mu=5, the stiffness is moderate, and avoiding
    # numerical Jacobian costs (which add to f call count) may reduce total calls.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOPRI5', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback to RK45 if DOPRI5 fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13
        )
        
    if not sol.success:
        # Last resort: Radau (original approach)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )

    y_end = sol.y[:, -1]
    
    if not np.all(np.isfinite(y_end)):
        # If non-finite, return zeros to avoid exception, though score will be 0
        return np.zeros_like(y0)
        
    return y_end
