import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    call_count = 0
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    sol = solve_ivp(
        f_wrapper, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-10, 
        atol=1e-12
    )
    
    if not sol.success:
        # Fallback to a more robust method if it fails, though unlikely
        sol = solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )

    return sol.y[:, -1]
