import numpy as np
import scipy.integrate as integrate_scipy
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    # Safety margin: 300s limit, aim to finish in 250s
    # solve_ivp doesn't easily support timeouts mid-integration, 
    # so we rely on the fact that 100-point ODEs are fast.
    
    # Use LSODA which automatically switches between stiff and non-stiff methods
    # Tight tolerances to minimize error
    sol = integrate_scipy.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='LSODA', 
        rtol=1e-10, 
        atol=1e-12,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to Radau if LSODA fails
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    if not sol.success:
        # Last resort: RK45
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-8, 
            atol=1e-10
        )

    return sol.y[:, -1]
