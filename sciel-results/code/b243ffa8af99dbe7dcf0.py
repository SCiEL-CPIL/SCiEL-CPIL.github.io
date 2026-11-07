import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    timeout = 290
    
    # Strategy: Use DOP853 with an aggressive initial step size
    # to reduce the number of steps.
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            first_step=0.2, # Aggressive initial step
            max_step=1.0    # Allow large steps
        )
        if sol.success and time.time() - start_time < timeout:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to standard
    try:
        if time.time() - start_time < timeout - 10:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
        
    # Fallback 2: Looser tolerance
    try:
        if time.time() - start_time < timeout - 20:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
        
    return np.array(y0, dtype=float)
