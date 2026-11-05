import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy.integrate.solve_ivp with LSODA method for robustness against stiffness.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Counter for function evaluations
    call_count = 0
    
    def f_wrapped(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    # Try LSODA first, which switches between Adams and BDF
    # Tight tolerances for high accuracy
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.01 # Prevents taking huge steps if accuracy allows, but LSODA is adaptive
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to BDF if LSODA fails
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Final fallback: RK45
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # If all fail, return initial state (should not happen)
    return y0
